import pu4c
import numpy as np
import os
from matplotlib.image import imread

datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')

def mmdet3d_nuscenes_infos():
    filepath = '/workspace/codevault/Det3D/mmdetection3d/data/nuscenes/nuscenes_infos_val.pkl'
    infos = pu4c.common.read_pickle(filepath)
    newinfos = {
        'metainfo': infos['metainfo'],
        'data_list': infos['data_list'][1:3],
    }
    datadb.set('mmdet3d/nuscenes_infos', newinfos)
def mmdet3d_kitti_000008():
    from mmdet3d.structures.ops.box_np_ops import box_camera_to_lidar
    filepath = '/workspace/codevault/Det3D/mmdetection3d/data/kitti/kitti_infos_val.pkl'
    data = pu4c.common.read_pickle(filepath)
    info = data['data_list'][5]
    cam_info = info['images']['CAM2']
    lidar_path = os.path.join('/datasets/KITTI/object/training/velodyne/', info['lidar_points']['lidar_path'])
    image_path = os.path.join('/datasets/KITTI/object/training/image_2/', cam_info['img_path'])
    points = pu4c.cv.read_points(lidar_path, num_features=4)
    image = imread(image_path)
    calib = {
        'lidar2img': np.array(cam_info['lidar2img']),
    }
    gt_boxes_camera = np.array([np.array(obj['bbox_3d']) for obj in info['instances'] if obj['bbox_label_3d'] != -1]) # 去掉 DontCare 目标
    gt_boxes_lidar = box_camera_to_lidar(gt_boxes_camera, np.array(info['images']['R0_rect']), np.array(info['lidar_points']['Tr_velo_to_cam']))
    gt_boxes_lidar[:, 2] += (gt_boxes_lidar[:, 5] / 2)
    boxes3d = gt_boxes_lidar
    labels = np.array([obj['bbox_label_3d'] for obj in info['instances'] if obj['bbox_label_3d'] != -1])
    datadb.set('mmdet3d/kitti/000008', [points, image, calib, boxes3d, labels]) 

def semantickitti_000000():
    from pu4c.config import semantickitti_learning_map as learning_map 
    from pu4c.config import semantickitti_classes as classes, semantickitti_colormap as colormap
    lidar_path = '/datasets/SemanticKITTI/dataset/sequences/00/velodyne/000000.bin'
    label_path = '/datasets/SemanticKITTI/dataset/sequences/00/labels/000000.label'
    points = pu4c.cv.read_points(lidar_path, num_features=4)
    labels = np.fromfile(label_path, dtype=np.uint32)
    labels = labels & 0xFFFF    # 等价于 % 2**16，取模后最大值为 255
    labels = np.vectorize(learning_map.__getitem__)(labels) # 映射完最大值 19
    labels = labels.astype(np.uint8)
    datadb.set("semantickitti/000000", data=[points, labels, classes, colormap])

def occ3d_nuscenes_scene0001_1e19d0():
    from pu4c.config import occ3d_classes as classes, occ3d_colormap as colormap
    label_path = "/datasets/nuScenes/Occ3D-nuScenes/gts/scene-0001/1e19d0a5189b46f4b62aa47508f2983e/labels.npz"
    data = np.load(label_path)
    labels, lidar_mask, camera_mask = data['semantics'], data['mask_lidar'], data['mask_camera']
    voxel_size, point_cloud_range = [0.4, 0.4, 0.4], np.array([-40, -40, -1, 40, 40, 5.4])
    
    FREE_LABEL = 17                     # 空体素类别
    valid_mask = labels != FREE_LABEL   # 非空体素掩膜，HWD
    valid_idx = np.where(valid_mask)    # 非空体素下标，tuple(X, Y, Z), X.shape(N=非空体素个数,)
    labels = labels[valid_idx]          # 非空体素的标签，(N,)，值域 0-16 共 17 类
    voxel_centers = np.array(valid_idx).T * voxel_size # 非空体素坐标即中心点坐标，(N, 3)，等于下标乘以体素大小
    voxel_centers = voxel_centers - ((point_cloud_range[3:6] - point_cloud_range[0:3]) / 2) # 坐标原点移动到场景中心

    colormap = np.array(colormap) / 255 # 归一化颜色值
    datadb.set("occ3d_nuscenes/scene0001-1e19d0", data=[voxel_centers, voxel_size, labels, classes, colormap])

def sklearn_digits():
    from sklearn import datasets
    digits = datasets.load_digits(n_class=6) # MNIST test set 的子集，10 类总样本数 1797, 特征维度 8x8=64
    data, label, (n_samples, n_features) = digits.data, digits.target, digits.data.shape
    datadb.set('sklearn/digits', [data, label])

if __name__ == '__main__':
    occ3d_nuscenes_scene0001_1e19d0()