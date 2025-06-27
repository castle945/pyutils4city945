import pu4c
import numpy as np
datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')


from pu4c.cv.utils import read_points
from pu4c.common.utils import read_pickle
import numpy as np
import os
infos = read_pickle("/workspace/files/blob/Det3DTrans/OpenPCDetTrans/data/kitti/kitti_infos_train.pkl")
info = infos[4]
points = read_points(filepath=f"/datasets/KITTI/object/training/velodyne/{info['point_cloud']['lidar_idx']}.bin")
boxes3d = info['annos']['gt_boxes_lidar']

map_name_to_id = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2}
cls_ids = np.array([map_name_to_id[name] for name in info['annos']['name'][:boxes3d.shape[0]]])
boxes3d_with_label = np.concatenate((boxes3d, cls_ids[:, None]), axis=1)
datadb.set("kitti/000010/points", data=[points, boxes3d, boxes3d_with_label])

import open3d as o3d
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(points[:, :3])
cloud = cloud.voxel_down_sample(0.5)
points_ds = np.array(cloud.points)
point_labels = np.concatenate((np.zeros(points.shape[0]), np.ones(points_ds.shape[0])), axis=0)
points = np.concatenate((points[:, :3], points_ds), axis=0)
datadb.set("kitti/000010/point_labels", data=[points, point_labels])


from sklearn import datasets
digits = datasets.load_digits(n_class=6) # MNIST test set 的子集，10 类总样本数 1797, 特征维度 8x8=64
data, label, (n_samples, n_features) = digits.data, digits.target, digits.data.shape
datadb.set('sklearn/digits', [data, label])


"""
occ3d-nus/labels.npz
    semantics: (grid_size_x=80/0.4=200, grid_size_y=80/0.4=200, grid_size_z=6.4/0.4=16), 每个体素的语义标签
    mask_lidar: grid_size_shape, 激光雷达视角下的掩膜，反射激光束的体素为激光点语义标签，激光束穿过的体素为空，其他体素为未知，掩膜标记未知体素为 0
    mask_camera: grid_size_shape, 同理相机视角下的掩膜
"""
nus_colormap = np.array([
    [0,   0,   0, 255],  # 0 undefined
    [255, 158, 0, 255],  # 1 car  orange
    [0, 0, 230, 255],    # 2 pedestrian  Blue
    [47, 79, 79, 255],   # 3 sign  Darkslategrey
    [220, 20, 60, 255],  # 4 CYCLIST  Crimson
    [255, 69, 0, 255],   # 5 traiffic_light  Orangered
    [255, 140, 0, 255],  # 6 pole  Darkorange
    [233, 150, 70, 255], # 7 construction_cone  Darksalmon
    [255, 61, 99, 255],  # 8 bycycle  Red
    [112, 128, 144, 255],# 9 motorcycle  Slategrey
    [222, 184, 135, 255],# 10 building Burlywood
    [0, 175, 0, 255],    # 11 vegetation  Green
    [165, 42, 42, 255],  # 12 trunk  nuTonomy green
    [0, 207, 191, 255],  # 13 curb, road, lane_marker, other_ground
    [75, 0, 75, 255],    # 14 walkable, sidewalk
    [128, 100, 100, 255], # 随便自定义的
    [128, 100, 100, 255], # 随便自定义的
    [255, 0, 0, 255],     # 17 unobsrvd
    [128, 128, 128, 255], # 18 for vis
], dtype=np.float32)
FREE_LABEL = 17 # 空体素类别
voxel_size = [0.4, 0.4, 0.4]
point_cloud_range = [-40, -40, -1, 40, 40, 5.4]
filepath = "/datasets/nuScenes/Occ3D-nuScenes/gts/scene-0001/1e19d0a5189b46f4b62aa47508f2983e/labels.npz"
data = np.load(filepath)
voxel_label, lidar_mask, camera_mask = data['semantics'], data['mask_lidar'], data['mask_camera']

# 获取稀疏体素中心点
valid_mask = voxel_label != FREE_LABEL # 非空体素掩膜
# grid_size_shape -> ((N=非空体素个数,), (N,), (N,)) 非空体素下标，即 valid_mask[valid_idx[0][i], valid_idx[1][i], valid_idx[2][i]] == True
valid_idx = np.where(valid_mask)
labels = voxel_label[valid_idx] # (N,) 非空体素的标签
voxel_centers = np.array(valid_idx).T * voxel_size # (3, N)->(N, 3) 稀疏体素下标乘以 voxel_size 得到稀疏体素中心点

colormap = nus_colormap[:, :3] / 255
datadb.set("occ3d-nuscenes/0001/voxel_semantic_mask", data=[voxel_centers, voxel_size, labels, colormap])

