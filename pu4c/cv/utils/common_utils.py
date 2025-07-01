import numpy as np
import os

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

def transform_matrix(rotation_mat: np.ndarray, translation: np.ndarray, inverse: bool = False) -> np.ndarray:
    """
    传入旋转矩阵和平移向量，返回变换矩阵或变换矩阵的逆  
    要求输入矩阵是行列式为 1 的 3×3 正交矩阵(即旋转矩阵的充要条件，一般可由欧拉角或四元数等转换而来)，而不能对任意矩阵求逆
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation_mat.T
        trans = np.transpose(-translation)
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation_mat
        tm[:3, 3] = np.transpose(translation)

    return tm
