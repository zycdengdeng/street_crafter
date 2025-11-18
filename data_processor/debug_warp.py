#!/usr/bin/env python3
"""调试拼接数据生成"""

import numpy as np
import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_rotation_matrix_z(angle_deg):
    """绕Z轴旋转矩阵"""
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    return R

def test_warp():
    scene_dir = '/home/zyc/street_crafter/data/waymo/049'
    frame = 0
    cam_id = 0

    # 读取
    rgb_path = f'{scene_dir}/images/{frame:06d}_{cam_id}.png'
    depth_path = f'{scene_dir}/lidar/dense_depth/{frame:06d}_{cam_id}_depth.npy'

    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth = np.load(depth_path)

    # 读取内参
    intrinsic_path = f'{scene_dir}/intrinsics/{cam_id}.txt'
    data = np.loadtxt(intrinsic_path)
    fx, fy, cx, cy = data[0], data[1], data[2], data[3]
    intrinsic = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ])

    print(f'Intrinsic:\\n{intrinsic}')

    h, w = rgb.shape[:2]
    mid = w // 2

    # 左半
    rgb_left = rgb[:, :mid]
    depth_left = depth[:, :mid]

    print(f'\\nProcessing left half: {rgb_left.shape}')

    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    print(f'fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}')

    # 反投影
    u, v = np.meshgrid(np.arange(mid), np.arange(h))
    valid = depth_left > 0

    u_valid = u[valid]
    v_valid = v[valid]
    z_valid = depth_left[valid]

    x = (u_valid - cx) * z_valid / fx
    y = (v_valid - cy) * z_valid / fy

    points_3d = np.stack([x, y, z_valid], axis=-1)
    colors = rgb_left[v_valid, u_valid]

    print(f'\\n3D points: {points_3d.shape}')
    print(f'3D range: x=[{points_3d[:, 0].min():.2f}, {points_3d[:, 0].max():.2f}]')
    print(f'          y=[{points_3d[:, 1].min():.2f}, {points_3d[:, 1].max():.2f}]')
    print(f'          z=[{points_3d[:, 2].min():.2f}, {points_3d[:, 2].max():.2f}]')

    # 旋转
    R = get_rotation_matrix_z(-22.5)
    points_3d_rot = points_3d @ R.T

    print(f'\\nAfter rotation:')
    print(f'3D range: x=[{points_3d_rot[:, 0].min():.2f}, {points_3d_rot[:, 0].max():.2f}]')
    print(f'          y=[{points_3d_rot[:, 1].min():.2f}, {points_3d_rot[:, 1].max():.2f}]')
    print(f'          z=[{points_3d_rot[:, 2].min():.2f}, {points_3d_rot[:, 2].max():.2f}]')

    # 重投影
    pixels = points_3d_rot @ intrinsic.T
    pixels = pixels[:, :2] / (pixels[:, 2:3] + 1e-8)

    print(f'\\nProjected pixels:')
    print(f'u range: [{pixels[:, 0].min():.1f}, {pixels[:, 0].max():.1f}]')
    print(f'v range: [{pixels[:, 1].min():.1f}, {pixels[:, 1].max():.1f}]')

    # 统计有效像素
    valid_proj = (
        (points_3d_rot[:, 2] > 0) &
        (pixels[:, 0] >= 0) & (pixels[:, 0] < mid) &
        (pixels[:, 1] >= 0) & (pixels[:, 1] < h)
    )

    print(f'\\nValid after projection: {valid_proj.sum()} / {len(valid_proj)}')

    if valid_proj.sum() == 0:
        print('ERROR: No valid points after projection!')
        return

    # 渲染
    warped = np.zeros((h, mid, 3), dtype=np.uint8)
    depth_buffer = np.full((h, mid), np.inf)

    valid_pixels = pixels[valid_proj].astype(np.int32)
    valid_depths = points_3d_rot[valid_proj, 2]
    valid_colors = colors[valid_proj]

    for i in range(len(valid_pixels)):
        u, v = valid_pixels[i]
        d = valid_depths[i]
        c = valid_colors[i]

        if 0 <= u < mid and 0 <= v < h and d < depth_buffer[v, u]:
            depth_buffer[v, u] = d
            warped[v, u] = c

    filled_pixels = (depth_buffer < np.inf).sum()
    print(f'\\nFilled pixels: {filled_pixels} / {h * mid}')

    # 保存
    output_path = '/home/zyc/street_crafter/data/waymo/049/test_warp_left.png'
    cv2.imwrite(output_path, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    print(f'\\nSaved to {output_path}')

    # 对比原图左半
    original_path = '/home/zyc/street_crafter/data/waymo/049/test_original_left.png'
    cv2.imwrite(original_path, cv2.cvtColor(rgb_left, cv2.COLOR_RGB2BGR))
    print(f'Original left saved to {original_path}')

if __name__ == '__main__':
    test_warp()
