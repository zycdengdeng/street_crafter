import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
from street_gaussian.config import cfg
from street_gaussian.utils.camera_utils import Camera
from street_gaussian.pointcloud_processor.base_processor import BasePointCloudProcessor
from street_gaussian.utils.graphics_utils import project_numpy
from data_processor.utils.box_utils import inbbox_points, bbox_to_corner3d
from data_processor.utils.pcd_utils import BasicPointCloud, fetchPly
from data_processor.utils.render_utils import render_pointcloud_diff_point_rasterization, render_pointcloud_pytorch3d
from data_processor.pandaset_processor.pandaset_helpers import load_camera_info, load_track, get_obj_info, \
    image_filename_to_cam, image_filename_to_frame, IMAGE_HEIGHT, IMAGE_WIDTH, CAM2NAME, NUM_FRAMES, PANDA_ID2CAMERA, PANDA_RIGID_DYNAMIC_CLASSES
from easyvolcap.utils.console_utils import *


class PandasetPointCloudProcessor(BasePointCloudProcessor):
    def __init__(self):
        print('Initializing Pandaset Point Cloud Processor')
        self.datadir = cfg.source_path
        self.delta_frames = cfg.data.delta_frames
        self.cams = cfg.data.cameras
        self.start_frame = cfg.data.selected_frames[0]
        self.end_frame = cfg.data.selected_frames[-1]

        # load calibration
        self.cam_poses, self.intrinsics = load_camera_info(self.datadir)

        # load tracks
        self.frame_instances, self.instances_info = load_track(self.datadir)

        # read lidar ply
        self.flip_axis = 1
        self.ply_dict = self.read_lidar_ply()

        # load timestamps
        timestamp_path = os.path.join(cfg.source_path, 'timestamps.json')
        with open(timestamp_path, 'r') as f:
            self.timestamps = json.load(f)

    def read_lidar_ply(self):
        ply_dict = dict()
        lidar_dir = os.path.join(self.datadir, 'lidar_forward')

        # read background ply (we only save the pointcloud which is visible to the input camera)
        lidar_background_dir = os.path.join(lidar_dir, 'background')
        ply_dict_background = dict()
        ply_dict_background_visible = dict()

        bkgd_ply_list = sorted([os.path.join(lidar_background_dir, f) for f in os.listdir(lidar_background_dir) if f.endswith('.ply') and f != 'full.ply'])
        bkgd_ply_list = [x for x in bkgd_ply_list if image_filename_to_frame(os.path.basename(x)) >= self.start_frame and image_filename_to_frame(os.path.basename(x)) <= self.end_frame]

        for i, bkgd_ply_path in enumerate(tqdm(bkgd_ply_list, desc='Reading Background ply')):
            bkgd_ply = fetchPly(bkgd_ply_path)
            frame = image_filename_to_frame(os.path.basename(bkgd_ply_path))

            mask = bkgd_ply.mask 
            xyz_world = bkgd_ply.points[mask]
            rgb = bkgd_ply.colors[mask]
            ply_dict_background[frame] = np.concatenate([xyz_world, rgb], axis=-1)
                
            visible_mask = np.zeros_like(xyz_world[..., 0], dtype=bool)
            for cam in self.cams:
                K = self.intrinsics[cam]
                RT = np.linalg.inv(self.cam_poses[frame, cam])
                _, visible_cam_mask = project_numpy(xyz_world, K, RT, IMAGE_HEIGHT, IMAGE_WIDTH)
                visible_mask = np.logical_or(visible_mask, visible_cam_mask)
            ply_dict_background_visible[frame] = visible_mask
        
        ply_dict['background'] = ply_dict_background
        ply_dict['background_visible'] = ply_dict_background_visible

        # read actor lidar
        lidar_actor_dir = os.path.join(lidar_dir, 'actor')
        for track_id in os.listdir(lidar_actor_dir):
            ply_dict_actor = dict()

            lidar_actor_dir_ = os.path.join(lidar_actor_dir, track_id)
            actor_ply_list = sorted([os.path.join(lidar_actor_dir_, f) for f in os.listdir(lidar_actor_dir_) if f.endswith('.ply') and f != 'full.ply'])
            for actor_ply_path in actor_ply_list:
                frame = image_filename_to_frame(os.path.basename(actor_ply_path))
                actor_ply = fetchPly(actor_ply_path)
                mask = actor_ply.mask

                if mask.sum() == 0:
                    continue

                xyz = actor_ply.points[mask]
                rgb = actor_ply.colors[mask]
                if self.instances_info[str(int(track_id))]['class_name'] in PANDA_RIGID_DYNAMIC_CLASSES:
                    num_pointcloud_1 = (xyz[:, self.flip_axis] > 0).sum()
                    num_pointcloud_2 = (xyz[:, self.flip_axis] < 0).sum()
                    if num_pointcloud_1 >= num_pointcloud_2:
                        xyz_part = xyz[xyz[:, self.flip_axis] > 0]
                        rgb_part = rgb[xyz[:, self.flip_axis] > 0]
                    else:
                        xyz_part = xyz[xyz[:, self.flip_axis] < 0]
                        rgb_part = rgb[xyz[:, self.flip_axis] < 0]
                    xyz_flip = xyz_part.copy()
                    xyz_flip[:, self.flip_axis] *= -1
                    rgb_flip = rgb_part.copy()
                    xyz = np.concatenate([xyz, xyz_flip], axis=0)
                    rgb = np.concatenate([rgb, rgb_flip], axis=0)

                ply_dict_actor[frame] = np.concatenate([xyz, rgb], axis=-1)

            ply_dict[track_id] = ply_dict_actor

        return ply_dict

    def transform_lidar_ply(self, lidar_ply, obj_info):  # type: ignore
        pose, box = obj_info['pose'], obj_info['box']
        xyz, rgb = lidar_ply[..., :3], lidar_ply[..., 3:]

        # filter points in bbox
        length, width, height = box
        bbox = np.array([[-length, -width, -height], [length, width, height]]) * 0.5
        corners3d = bbox_to_corner3d(bbox)
        inbbox_mask = inbbox_points(xyz, corners3d)
        xyz = xyz[inbbox_mask]
        rgb = rgb[inbbox_mask]

        xyz_homo = np.concatenate([xyz, np.ones_like(xyz[..., :1])], axis=-1)
        xyz = (xyz_homo @ pose.T)[..., :3]
        lidar_ply = np.concatenate([xyz, rgb], axis=-1)
        return lidar_ply

    @torch.no_grad
    def render_condition(self, camera: Camera, objects_info):
        rgb_save_path = camera.meta['guidance_rgb_path']
        mask_save_path = camera.meta['guidance_mask_path']

        if os.path.exists(rgb_save_path) and os.path.exists(mask_save_path) and not cfg.diffusion.force_render_condition:
            # camera.guidance['mask'] = (PILtoTorch(Image.open(mask_save_path)) > 0.5)[None, ..., 0]
            return

        c2w = camera.get_extrinsic().cpu().numpy()
        ixt = camera.get_intrinsic().cpu().numpy()
        h, w = camera.image_height, camera.image_width

        # set actor ids
        frame = camera.meta['frame']
        start_frame = max(self.start_frame, frame - self.delta_frames)
        end_frame = min(self.end_frame, frame + self.delta_frames)
        actor_ids = []
        for actor in objects_info.values():
            if frame >= actor['start_frame'] and frame <= actor['end_frame']:
                actor_ids.append(actor['track_id'])

        ply_frame_dict = self.make_lidar_ply(start_frame=start_frame, end_frame=end_frame, actor_ids=actor_ids)
        ply_frame = [ply_frame_dict.pop('background')]

        cam_name = CAM2NAME[camera.meta['cam']]
        cam_timestamps = self.timestamps[cam_name]
        box_timestamps = self.timestamps[PANDA_ID2CAMERA[0]]
        for track_id, ply_actor_frame in ply_frame_dict.items():
            frame_annotations = self.instances_info[str(int(track_id))]['frame_annotations']
            timestamp = cam_timestamps[frame]
            obj_info = get_obj_info(frame_annotations=frame_annotations, box_timestamps=box_timestamps, timestamp=timestamp)
            if obj_info is None:
                continue

            ply_actor_frame = self.transform_lidar_ply(ply_actor_frame, obj_info)
            ply_frame.append(ply_actor_frame)

        ply_frame = np.concatenate(ply_frame, axis=0)
        ply_xyz, ply_rgb = ply_frame[..., :3], ply_frame[..., 3:]
        ply_mask = np.ones_like(ply_xyz[..., 0]).astype(bool)
        ply_feature = np.concatenate([ply_rgb, ply_mask[:, None]], axis=-1)
        ply_render = render_pointcloud_diff_point_rasterization(
            c2w[None], ixt[None], ply_xyz[None], ply_feature[None], h, w,
            scale=cfg.render.scale,
            use_ndc_scale=cfg.render.use_ndc_scale,
            use_knn_scale=cfg.render.use_knn_scale,
        )     
        # ply_render = render_pointcloud_pytorch3d(c2w[None], ixt[None], ply_xyz[None], ply_feature[None], h, w)
        ply_render_result_rgb, ply_render_result_mask = ply_render[0, ..., :3], ply_render[0, ..., 3:4]
        ply_render_result_rgb = (ply_render_result_rgb.cpu().numpy() * 255).astype(np.uint8)
        ply_render_result_mask = (ply_render_result_mask.cpu().numpy() * 255).astype(np.uint8)
        save_dir = os.path.dirname(rgb_save_path)
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(rgb_save_path, ply_render_result_rgb[..., [2, 1, 0]])
        cv2.imwrite(mask_save_path, ply_render_result_mask)
