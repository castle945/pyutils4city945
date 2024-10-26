import numpy as np
import os
import copy


def read_points(filepath, num_features=4, transmat=None):
    """
    Args:
        transmat: 4x4 变换矩阵，某些点云可能希望直接进行坐标变换
    """
    filetype = os.path.splitext(filepath)[-1]
    if filetype == ".bin":
        points = np.fromfile(filepath, dtype=np.float32).reshape(-1, num_features)
    elif filetype == ".pcd" or filetype == '.ply':
        import open3d as o3d
        points = np.asarray(o3d.io.read_point_cloud(filepath).points)
    elif filetype == ".npy":
        points = np.load(filepath)
    elif filetype == ".pkl" or filetype == ".gz": # '.pkl.gz'
        import pandas as pd
        points = pd.read_pickle(filepath).to_numpy()
    elif filetype == ".txt":
        points = np.loadtxt(filepath, dtype=np.float32).reshape(-1, num_features)
    else:
        raise TypeError("unsupport file type")

    if transmat is not None:
        points[:, :3] = (transmat[:3, :3] @ points[:, :3].T +  transmat[:3, [3]]).T

    return points

def transform_matrix(rotation_mat, translation, inverse: bool = False) -> np.ndarray:
    """
    返回变换矩阵或变换矩阵的逆，要求传入的旋转矩阵和平移向量必须是有实际意义的(即从 3x4 矩阵中拿到的)，而不能对任意矩阵求逆
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation_mat.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation_mat
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm
def limit_period(val,
                 offset: float = 0.5,
                 period: float = np.pi):
    """Limit the value into a period for periodic function.
    周期函数的区间映射，映射到 [-offset*period, (1-offset) * period]，floor 取小于输入 x 的最大整数
    常用于目标检测中规范化偏航角，坐标变换后的偏航角可能为n*[0,2pi]，以 2pi 为周期，可以将其调整为[-pi,pi]    
    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
    limited_val = val - np.floor(val / period + offset) * period
    return limited_val

def boxes3d_to_corners(boxes3d):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    dims = len(boxes3d.shape)
    if dims == 1:
        boxes3d = boxes3d[np.newaxis, :]

    template = np.array([
        [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1],
        [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
    ]) / 2.0

    corners3d = boxes3d[:, None, 3:6].repeat(8, 1).reshape(-1, 8, 3) * template
    corners3d = rotate_points_along_z(corners3d, boxes3d[:, 6])
    corners3d += boxes3d[:, None, 0:3]

    return corners3d[0] if dims == 1 else corners3d
def corners_to_boxes3d(corners):
    corners = copy.deepcopy(corners)
    centers = np.mean(corners, axis=1)
    corners -= centers[:, np.newaxis, :]
    dims = np.vstack([
        np.linalg.norm(corners[:, 1] - corners[:, 0], axis=-1),
        np.linalg.norm(corners[:, 2] - corners[:, 0], axis=-1),
        np.linalg.norm(corners[:, 3] - corners[:, 0], axis=-1),
    ]).T
    yaw = np.arctan2(corners[:, 1, 1] - corners[:, 0, 1], corners[:, 1, 0] - corners[:, 0, 0])
    return np.hstack([centers, dims, yaw[:, np.newaxis]])

def rotate_points_along_z(points, angle):
    cosa = np.cos(angle)
    sina = np.sin(angle)
    zeros = np.zeros(points.shape[0])
    ones = np.ones(points.shape[0])
    rot_matrix = np.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), axis=1).reshape(-1, 3, 3)
    points_rot = points[:, :, :3] @ rot_matrix
    return points_rot
def mask_points_and_boxes_outside_range(points, limit_range, boxes3d=None):
    point_mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    box_mask = ((boxes3d[:, :3] >= limit_range[:3]) & (boxes3d[:, :3]  <= limit_range[3:6])).all(axis=-1) if boxes3d is not None else None

    return point_mask, box_mask
def get_oriented_bounding_box_corners(xyz, lwh, axis_angles):
    """
    轴角转旋转矩阵（暂只考虑偏航）来将其旋转为有向包围盒，计算盒子的 8 个角点，添加连线
    Locals:
        lines: (10, 2), 预定义的 14 条连线
              4 -------- 6
             /|         /|
            5 -------- 3 .
            | |        | |
            . 7 -------- 1          
            |/         |/       z |/ x  
            2 -------- 0      y - 0
    Returns:
        corners: (N, 8, 3)
    """
    x, y, z = xyz
    l, w, h = lwh
    roll, pitch, yaw = axis_angles
    xdif, ydif, zdif = l/2, w/2, h/2
    offsets = np.array([
        [-xdif,  xdif, -xdif, -xdif, xdif, -xdif,  xdif,  xdif],
        [-ydif, -ydif,  ydif, -ydif, ydif,  ydif, -ydif,  ydif],
        [-zdif, -zdif, -zdif,  zdif, zdif,  zdif,  zdif, -zdif],
    ])
    R_x = np.array([
        [ 1, 0            ,  0          ],
        [ 0, np.cos(roll) , -np.sin(roll)],
        [ 0, np.sin(roll) ,  np.cos(roll)],
    ])
    R_y = np.array([
        [ np.cos(pitch),  0,  np.sin(pitch)],
        [ 0            ,  1,  0            ],
        [-np.sin(pitch),  0,  np.cos(pitch)],
    ])
    R_z = np.array([
        [ np.cos(yaw), -np.sin(yaw),  0],
        [ np.sin(yaw),  np.cos(yaw),  0],
        [ 0          ,  0          ,  1],
    ])
    R = R_x @ R_y @ R_z
    corners = (R @ offsets + np.array([[x], [y], [z]])).T
    
    return corners
def get_oriented_bounding_box_lines(head_cross_lines=True):
    lines = [
                [0, 2], [0, 3], [2, 5], [3, 5],
                [0, 1], [3, 6], [5, 4], [2, 7],
                [1, 6], [1, 7], [7, 4], [4, 6],
            ]
    if head_cross_lines:
        lines.extend([[1, 4], [6, 7]])
    return lines

