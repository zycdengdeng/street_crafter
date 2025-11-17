import torch
import torch.nn as nn
import numpy as np
from street_gaussian.utils.general_utils import get_expon_lr_func, quaternion_slerp, quaternion_raw_multiply_theta
from street_gaussian.config import cfg
from street_gaussian.utils.camera_utils import Camera
# fmt: off
from easyvolcap.utils.console_utils import *
# fmt: on


class ActorPose(nn.Module):
    def __init__(self, camera_tracklets, camera_timestamps, obj_info):
        # tracklets: [cams, num_frames, max_obj, [x, y, z, qw, qx, qy, qz, valid]]
        super().__init__()
        self.camera_timestamps = camera_timestamps
        camera_tracklets = torch.from_numpy(camera_tracklets).float().cuda()
        self.valid_mask = camera_tracklets[..., -1].int()  # [num_cams, num_frames, max_obj]
        self.input_trans = camera_tracklets[..., :3]
        self.input_rots = camera_tracklets[..., 3:7]

        self.opt_track = cfg.model.nsg.opt_track
        if self.opt_track:
            self.opt_trans = nn.Parameter(torch.zeros_like(self.input_trans)).requires_grad_(True)
            # [num_cams, num_frames, max_obj, [dx, dy, dz]]

            self.opt_rots = nn.Parameter(torch.zeros_like(self.input_rots[..., :1])).requires_grad_(True)
            # [num_cams, num_frames, max_obj, [dtheta]

        self.obj_info = obj_info

    def save_state_dict(self, is_final):
        state_dict = dict()
        if self.opt_track:
            state_dict['params'] = self.state_dict()
            if not is_final:
                state_dict['optimizer'] = self.optimizer.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        if self.opt_track:
            super().load_state_dict(state_dict['params'])
            if cfg.mode == 'train' and 'optimizer' in state_dict:
                self.optimizer.load_state_dict(state_dict['optimizer'])

    def training_setup(self):
        args = cfg.optim
        if self.opt_track:
            params = [
                {'params': [self.opt_trans], 'lr': args.track_position_lr_init, 'name': 'opt_trans'},
                {'params': [self.opt_rots], 'lr': args.track_rotation_lr_init, 'name': 'opt_rots'},
            ]

            self.opt_trans_scheduler_args = get_expon_lr_func(lr_init=args.track_position_lr_init,
                                                              lr_final=args.track_position_lr_final,
                                                              lr_delay_mult=args.track_position_lr_delay_mult,
                                                              max_steps=args.track_position_max_steps,
                                                              warmup_steps=args.opacity_reset_interval)

            self.opt_rots_scheduler_args = get_expon_lr_func(lr_init=args.track_rotation_lr_init,
                                                             lr_final=args.track_rotation_lr_final,
                                                             lr_delay_mult=args.track_rotation_lr_delay_mult,
                                                             max_steps=args.track_rotation_max_steps,
                                                             warmup_steps=args.opacity_reset_interval)

            self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-15)

    def update_learning_rate(self, iteration):
        if self.opt_track:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "opt_trans":
                    lr = self.opt_trans_scheduler_args(iteration)
                    param_group['lr'] = lr
                if param_group["name"] == "opt_rots":
                    lr = self.opt_rots_scheduler_args(iteration)
                    param_group['lr'] = lr

    def update_optimizer(self):
        if self.opt_track:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def get_tracking_translation_(self, cam, frame_idx, id, is_lidar_pose: bool = False):
        if self.opt_track and not is_lidar_pose:
            return self.input_trans[cam, frame_idx, id] + self.opt_trans[cam, frame_idx, id]
        else:
            return self.input_trans[cam, frame_idx, id]

    def get_tracking_rotation_(self, cam, frame_idx, id, is_lidar_pose: bool = False):
        if self.opt_track and not is_lidar_pose:
            return quaternion_raw_multiply_theta(self.input_rots[cam, frame_idx, id], self.opt_rots[cam, frame_idx, id])
        else:
            return self.input_rots[cam, frame_idx, id]

    def get_tracking_translation(self, object_id, camera: Camera):
        cam = camera.meta['cam']
        frame_idx = camera.meta['frame_idx']
        id = self.obj_info[object_id]['id']
        assert self.valid_mask[cam, frame_idx, id] == 1, 'Invalid object'

        need_interpolate = self.opt_track and \
            camera.meta['is_val'] and \
            frame_idx > 0 and \
            frame_idx < self.valid_mask.shape[1] - 1 and \
            self.valid_mask[cam, frame_idx - 1, id] == 1 and \
            self.valid_mask[cam, frame_idx + 1, id] == 1

        if need_interpolate:
            pre_trans = self.get_tracking_translation_(cam, frame_idx - 1, id, camera.meta['is_novel_view'])
            pre_timestamp = self.camera_timestamps[cam][frame_idx - 1]
            next_trans = self.get_tracking_translation_(cam, frame_idx + 1, id, camera.meta['is_novel_view'])
            next_timestamp = self.camera_timestamps[cam][frame_idx + 1]
            cur_timestamp = camera.meta['timestamp']
            alpha = (cur_timestamp - pre_timestamp) / (next_timestamp - pre_timestamp)
            trans = alpha * next_trans + (1. - alpha) * pre_trans
        else:
            trans = self.get_tracking_translation_(cam, frame_idx, id, camera.meta['is_novel_view'])
        return trans

    def get_tracking_rotation(self, object_id, camera: Camera):
        cam = camera.meta['cam']
        frame_idx = camera.meta['frame_idx']
        id = self.obj_info[object_id]['id']
        # print(self.valid_mask.unique())
        assert self.valid_mask[cam, frame_idx, id] == 1, 'Invalid object'
        need_interpolate = self.opt_track and \
            camera.meta['is_val'] and \
            frame_idx > 0 and \
            frame_idx < self.valid_mask.shape[1] - 1 and \
            self.valid_mask[cam, frame_idx - 1, id] == 1 and \
            self.valid_mask[cam, frame_idx + 1, id] == 1

        if need_interpolate:
            pre_rots = self.get_tracking_rotation_(cam, frame_idx - 1, id, camera.meta['is_novel_view'])
            pre_timestamp = self.camera_timestamps[cam][frame_idx - 1]
            next_rots = self.get_tracking_rotation_(cam, frame_idx + 1, id, camera.meta['is_novel_view'])
            next_timestamp = self.camera_timestamps[cam][frame_idx + 1]
            cur_timestamp = camera.meta['timestamp']
            alpha = (cur_timestamp - pre_timestamp) / (next_timestamp - pre_timestamp)
            rots = quaternion_slerp(pre_rots, next_rots, alpha)
        else:
            rots = self.get_tracking_rotation_(cam, frame_idx, id, camera.meta['is_novel_view'])

        return rots
