#!/usr/bin/env python3
"""
生成拼接训练数据（方案B2：3D深度引导变换）

输入：主视角CAM0的RGB和深度
输出：
  - stitched_rgb: 拼接后的RGB
  - stitched_depth: 外插视角的深度（重新渲染）
  - gt_rgb: 原始CAM0真值
"""

import os
import sys
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.pcd_utils import fetchPly


def load_calibration(datadir):
    """加载相机内外参"""
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    extrinsics = {}
    intrinsics = {}

    for cam_file in os.listdir(extrinsics_dir):
        cam_id = int(cam_file.split('.')[0])
        ext_path = os.path.join(extrinsics_dir, cam_file)
        extrinsics[cam_id] = np.loadtxt(ext_path).reshape(4, 4)

    for cam_file in os.listdir(intrinsics_dir):
        cam_id = int(cam_file.split('.')[0])
        int_path = os.path.join(intrinsics_dir, cam_file)
        # 内参格式: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
        data = np.loadtxt(int_path)
        fx, fy, cx, cy = data[0], data[1], data[2], data[3]
        intrinsics[cam_id] = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ])

    return extrinsics, intrinsics


def load_ego_poses(datadir):
    """加载车辆位姿"""
    ego_pose_dir = os.path.join(datadir, 'ego_pose')

    ego_poses = {}
    for pose_file in sorted(os.listdir(ego_pose_dir)):
        frame = int(pose_file.split('.')[0])
        pose_path = os.path.join(ego_pose_dir, pose_file)
        ego_poses[frame] = np.loadtxt(pose_path).reshape(4, 4)

    return ego_poses


def get_rotation_matrix_z(angle_deg):
    """绕Z轴旋转矩阵"""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    R = np.array([
        [cos_a, -sin_a, 0, 0],
        [sin_a, cos_a, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return R


def unproject_depth_to_3d(depth, intrinsic, mask=None):
    """
    深度图反投影到3D点云

    Args:
        depth: [H, W] 深度图
        intrinsic: [3, 3] 相机内参
        mask: [H, W] 有效区域mask（可选）

    Returns:
        points_3d: [N, 3] 3D点云
        colors: [N, 3] 对应颜色（需要外部传入）
        pixel_coords: [N, 2] 原始像素坐标
    """
    h, w = depth.shape
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    # 生成像素网格
    u, v = np.meshgrid(np.arange(w), np.arange(h))

    # 应用mask
    if mask is not None:
        valid = (depth > 0) & mask
    else:
        valid = depth > 0

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = depth[valid]

    # 反投影
    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy
    z = z_valid

    points_3d = np.stack([x, y, z], axis=-1)  # [N, 3]
    pixel_coords = np.stack([u_valid, v_valid], axis=-1)  # [N, 2]

    return points_3d, pixel_coords


def project_3d_to_image(points_3d, intrinsic, img_h, img_w):
    """
    3D点云投影到图像

    Args:
        points_3d: [N, 3] 相机坐标系下的3D点
        intrinsic: [3, 3] 相机内参
        img_h, img_w: 图像尺寸

    Returns:
        pixel_coords: [N, 2] 像素坐标
        valid_mask: [N] 是否在图像范围内
    """
    # 投影
    pixels = points_3d @ intrinsic.T  # [N, 3]
    pixels = pixels[:, :2] / (pixels[:, 2:3] + 1e-8)  # [N, 2]

    # 筛选有效点
    valid_mask = (
        (points_3d[:, 2] > 0) &
        (pixels[:, 0] >= 0) & (pixels[:, 0] < img_w) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < img_h)
    )

    return pixels, valid_mask


def warp_image_with_depth(rgb, depth, intrinsic, rotation_angle, side='left'):
    """
    使用深度做3D引导的图像变换

    Args:
        rgb: [H, W, 3] RGB图像（左半或右半）
        depth: [H, W] 深度图
        intrinsic: [3, 3] 相机内参
        rotation_angle: 旋转角度（度）
        side: 'left' 或 'right'

    Returns:
        warped_rgb: [H, W, 3] 变换后的RGB
        warped_depth: [H, W] 变换后的深度
    """
    h, w = rgb.shape[:2]

    # 1. 反投影到3D
    points_3d_cam, pixel_coords = unproject_depth_to_3d(depth, intrinsic)

    # 获取颜色
    colors = rgb[pixel_coords[:, 1], pixel_coords[:, 0]]  # [N, 3]

    # 2. 应用旋转变换
    R = get_rotation_matrix_z(rotation_angle)[:3, :3]
    points_3d_rotated = points_3d_cam @ R.T  # [N, 3]

    # 3. 重投影到2D
    new_pixels, valid_mask = project_3d_to_image(
        points_3d_rotated, intrinsic, h, w
    )

    # 4. 渲染到图像（使用z-buffer）
    warped_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    warped_depth = np.zeros((h, w), dtype=np.float32)
    depth_buffer = np.full((h, w), np.inf, dtype=np.float32)

    valid_points = points_3d_rotated[valid_mask]
    valid_pixels = new_pixels[valid_mask].astype(np.int32)
    valid_colors = colors[valid_mask]
    valid_depths = valid_points[:, 2]

    for i in range(len(valid_pixels)):
        u, v = valid_pixels[i]
        d = valid_depths[i]

        if 0 <= u < w and 0 <= v < h and d < depth_buffer[v, u]:
            depth_buffer[v, u] = d
            warped_rgb[v, u] = valid_colors[i]
            warped_depth[v, u] = d

    return warped_rgb, warped_depth


def read_lidar_ply(lidar_dir, ego_poses, start_frame, end_frame):
    """读取并聚合多帧点云（复用waymo_render_lidar_pcd.py的逻辑）"""
    lidar_background_dir = os.path.join(lidar_dir, 'background')

    aggregated_xyz = []
    aggregated_rgb = []

    for frame in range(start_frame, end_frame + 1):
        ply_path = os.path.join(lidar_background_dir, f'{frame:06d}.ply')

        if not os.path.exists(ply_path):
            continue

        ply_data = fetchPly(ply_path)
        mask = ply_data.mask

        if mask.sum() == 0:
            continue

        xyz_vehicle = ply_data.points[mask]
        rgb = ply_data.colors[mask]

        # 转换到世界坐标系
        xyz_vehicle_homo = np.concatenate([xyz_vehicle, np.ones((xyz_vehicle.shape[0], 1))], axis=-1)
        xyz_world = (xyz_vehicle_homo @ ego_poses[frame].T)[:, :3]

        aggregated_xyz.append(xyz_world)
        aggregated_rgb.append(rgb)

    if len(aggregated_xyz) == 0:
        return None, None

    aggregated_xyz = np.concatenate(aggregated_xyz, axis=0)
    aggregated_rgb = np.concatenate(aggregated_rgb, axis=0)

    return aggregated_xyz, aggregated_rgb


def render_interpolated_depth(scene_dir, frame, cam_id, rotation_angle,
                               ego_poses, extrinsics, intrinsics,
                               delta_frames=10):
    """
    渲染外插视角的深度图

    Args:
        scene_dir: 场景目录
        frame: 当前帧
        cam_id: 相机ID
        rotation_angle: 外插角度
        ego_poses: 车辆位姿
        extrinsics: 相机外参
        intrinsics: 相机内参
        delta_frames: 聚合帧数

    Returns:
        depth_map: [H, W] 外插深度图
    """
    # 读取点云
    lidar_dir = os.path.join(scene_dir, 'lidar')
    num_frames = len(ego_poses)

    start_frame = max(0, frame - delta_frames)
    end_frame = min(num_frames - 1, frame + delta_frames)

    xyz_world, rgb_world = read_lidar_ply(lidar_dir, ego_poses, start_frame, end_frame)

    if xyz_world is None:
        return None

    # 计算外插视角的相机位姿
    ego_pose = ego_poses[frame]

    # 在相机坐标系下旋转
    R_interp = get_rotation_matrix_z(rotation_angle)
    extrinsic_interp = extrinsics[cam_id] @ R_interp

    c2w = ego_pose @ extrinsic_interp
    w2c = np.linalg.inv(c2w)

    # 投影点云
    intrinsic = intrinsics[cam_id]

    # 获取图像尺寸（从第一张图像）
    sample_img_path = os.path.join(scene_dir, 'images', f'{frame:06d}_{cam_id}.png')
    sample_img = cv2.imread(sample_img_path)
    if sample_img is None:
        return None
    img_h, img_w = sample_img.shape[:2]

    # 转换到相机坐标系
    xyz_homo = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1))], axis=-1)
    xyz_cam = (xyz_homo @ w2c.T)[:, :3]

    # 投影
    depth = xyz_cam[:, 2]
    pixel = xyz_cam @ intrinsic.T
    pixel = pixel[:, :2] / (pixel[:, 2:] + 1e-8)

    # 筛选有效点
    valid_mask = (
        (depth > 0.1) &
        (pixel[:, 0] >= 0) & (pixel[:, 0] < img_w) &
        (pixel[:, 1] >= 0) & (pixel[:, 1] < img_h)
    )

    valid_pixel = pixel[valid_mask].astype(np.int32)
    valid_depth = depth[valid_mask]

    # 渲染深度图（z-buffer）
    depth_map = np.zeros((img_h, img_w), dtype=np.float32)
    depth_buffer = np.full((img_h, img_w), np.inf, dtype=np.float32)

    for i in range(len(valid_pixel)):
        u, v = valid_pixel[i]
        d = valid_depth[i]

        if 0 <= u < img_w and 0 <= v < img_h and d < depth_buffer[v, u]:
            depth_buffer[v, u] = d
            depth_map[v, u] = d

    return depth_map


def generate_stitched_data(scene_dir, output_dir, cam_id=0,
                           rotation_angle=22.5, delta_frames=10):
    """
    生成拼接训练数据

    Args:
        scene_dir: 场景目录
        output_dir: 输出目录
        cam_id: 相机ID
        rotation_angle: 旋转角度（一半的角度，左右各这么多）
        delta_frames: 渲染深度时聚合的帧数
    """
    print(f"Processing scene: {scene_dir}")

    # 加载标定数据
    extrinsics, intrinsics = load_calibration(scene_dir)
    ego_poses = load_ego_poses(scene_dir)

    intrinsic = intrinsics[cam_id]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取帧数
    images_dir = os.path.join(scene_dir, 'images')
    dense_depth_dir = os.path.join(scene_dir, 'lidar', 'dense_depth')

    if not os.path.exists(dense_depth_dir):
        print(f"Error: Dense depth not found at {dense_depth_dir}")
        print("Please run waymo_render_lidar_pcd.py first to generate dense depth maps.")
        return

    # 获取所有帧
    image_files = sorted([f for f in os.listdir(images_dir) if f.endswith(f'_{cam_id}.png')])
    num_frames = len(image_files)

    print(f"Total frames: {num_frames}")

    for idx in tqdm(range(num_frames), desc='Generating stitched data'):
        frame = idx

        # 读取RGB和深度
        rgb_path = os.path.join(images_dir, f'{frame:06d}_{cam_id}.png')
        depth_path = os.path.join(dense_depth_dir, f'{frame:06d}_{cam_id}_depth.npy')

        if not os.path.exists(rgb_path) or not os.path.exists(depth_path):
            continue

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = np.load(depth_path)

        h, w = rgb.shape[:2]
        mid = w // 2

        # 拆分左右
        rgb_left = rgb[:, :mid]
        rgb_right = rgb[:, mid:]
        depth_left = depth[:, :mid]
        depth_right = depth[:, mid:]

        # 调整内参（因为crop了）
        intrinsic_left = intrinsic.copy()
        intrinsic_right = intrinsic.copy()
        intrinsic_right[0, 2] -= mid  # cx偏移

        # 3D变换
        warped_left, _ = warp_image_with_depth(
            rgb_left, depth_left, intrinsic_left, -rotation_angle, 'left'
        )
        warped_right, _ = warp_image_with_depth(
            rgb_right, depth_right, intrinsic_right, rotation_angle, 'right'
        )

        # 拼接RGB
        stitched_rgb = np.concatenate([warped_left, warped_right], axis=1)

        # 渲染外插深度
        stitched_depth = render_interpolated_depth(
            scene_dir, frame, cam_id, 0,  # 外插视角在中间，旋转角度为0（相对于左右的中点）
            ego_poses, extrinsics, intrinsics, delta_frames
        )

        # 保存
        stitched_rgb_path = os.path.join(output_dir, f'{frame:06d}_{cam_id}_stitched.png')
        stitched_depth_path = os.path.join(output_dir, f'{frame:06d}_{cam_id}_stitched_depth.npy')
        gt_rgb_path = os.path.join(output_dir, f'{frame:06d}_{cam_id}_gt.png')

        cv2.imwrite(stitched_rgb_path, cv2.cvtColor(stitched_rgb, cv2.COLOR_RGB2BGR))
        if stitched_depth is not None:
            np.save(stitched_depth_path, stitched_depth)
        cv2.imwrite(gt_rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        # 可选：保存深度可视化
        if idx % 10 == 0 and stitched_depth is not None:
            depth_vis = stitched_depth.copy()
            valid_mask = depth_vis > 0
            if valid_mask.any():
                depth_vis[valid_mask] = (depth_vis[valid_mask] - depth_vis[valid_mask].min()) / \
                                        (depth_vis[valid_mask].max() - depth_vis[valid_mask].min() + 1e-8)
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
                depth_vis[~valid_mask] = 0
                depth_vis_path = os.path.join(output_dir, f'{frame:06d}_{cam_id}_stitched_depth_vis.png')
                cv2.imwrite(depth_vis_path, depth_vis)

    print(f"Done! Output saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate stitched training data for view interpolation')
    parser.add_argument('--scene_dir', type=str, required=True,
                        help='Scene directory (e.g., /path/to/waymo/049)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (default: scene_dir/stitched_training)')
    parser.add_argument('--cam_id', type=int, default=0,
                        help='Camera ID (default: 0)')
    parser.add_argument('--rotation_angle', type=float, default=22.5,
                        help='Rotation angle for each half (default: 22.5 degrees)')
    parser.add_argument('--delta_frames', type=int, default=10,
                        help='Number of frames to aggregate for depth rendering (default: 10)')

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.scene_dir, 'stitched_training')

    generate_stitched_data(
        scene_dir=args.scene_dir,
        output_dir=args.output_dir,
        cam_id=args.cam_id,
        rotation_angle=args.rotation_angle,
        delta_frames=args.delta_frames
    )


if __name__ == '__main__':
    main()
