# rpc_server_ip, rpc_server_port = "192.10.84.159", 23207 # 如果有公网 IP
rpc_server_ip, rpc_server_port = "127.0.0.1", 30570 # 如果没有则通过 SSH 转发
cache_dir = "/tmp/pu4c"
cache_pkl = '/workspace/.cache/cache.pkl' # 可用于在两个终端中传递数据

semantickitti_learning_map = {
    0 : 0,     # "unlabeled"
    1 : 0,     # "outlier" mapped to "unlabeled" --------------------------mapped
    10: 1,     # "car"
    11: 2,     # "bicycle"
    13: 5,     # "bus" mapped to "other-vehicle" --------------------------mapped
    15: 3,     # "motorcycle"
    16: 5,     # "on-rails" mapped to "other-vehicle" ---------------------mapped
    18: 4,     # "truck"
    20: 5,     # "other-vehicle"
    30: 6,     # "person"
    31: 7,     # "bicyclist"
    32: 8,     # "motorcyclist"
    40: 9,     # "road"
    44: 10,    # "parking"
    48: 11,    # "sidewalk"
    49: 12,    # "other-ground"
    50: 13,    # "building"
    51: 14,    # "fence"
    52: 0,     # "other-structure" mapped to "unlabeled" ------------------mapped
    60: 9,     # "lane-marking" to "road" ---------------------------------mapped
    70: 15,    # "vegetation"
    71: 16,    # "trunk"
    72: 17,    # "terrain"
    80: 18,    # "pole"
    81: 19,    # "traffic-sign"
    99: 0,     # "other-object" to "unlabeled" ----------------------------mapped
    252: 1,    # "moving-car" to "car" ------------------------------------mapped
    253: 7,    # "moving-bicyclist" to "bicyclist" ------------------------mapped
    254: 6,    # "moving-person" to "person" ------------------------------mapped
    255: 8,    # "moving-motorcyclist" to "motorcyclist" ------------------mapped
    256: 5,    # "moving-on-rails" mapped to "other-vehicle" --------------mapped
    257: 5,    # "moving-bus" mapped to "other-vehicle" -------------------mapped
    258: 4,    # "moving-truck" to "truck" --------------------------------mapped
    259: 5,    # "moving-other"-vehicle to "other-vehicle" ----------------mapped
}
semantickitti_classes = [
    'unlabeled',
    'car', 'bicycle', 'motorcycle', 'truck', 'bus', 'person',
    'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
    'other-ground', 'building', 'fence', 'vegetation',
    'trunck', 'terrian', 'pole', 'traffic-sign'
]
semantickitti_colormap = [
    [255, 255, 255],
    [100, 150, 245], [100, 230, 245], [30, 60, 150],
    [80, 30, 180], [100, 80, 250], [155, 30, 30],
    [255, 40, 200], [150, 30, 90], [255, 0, 255],
    [255, 150, 255], [75, 0, 75], [175, 0, 75], [255, 200, 0],
    [255, 120, 50], [0, 175, 0], [135, 60, 0], [150, 240, 80],
    [255, 240, 150], [255, 0, 0]
]

# copy from https://github.com/Tsinghua-MARS-Lab/CVT-Occ/blob/8291052c9bdb402e4706bed842e3ec9fd28e0bf5/projects/configs/cvtocc/cvtocc_nuscenes.py
occ3d_classes = [
    'others',               # 也即被占用未知类别的通用物体 GO
    'barrier',
    'bicycle',
    'bus',
    'car',
    'construction_vehicle',
    'motorcycle',
    'pedestrian',
    'traffic_cone',
    'trailer',
    'truck',
    'driveable_surface',    # 路面
    'other_flat',
    'sidewalk',
    'terrain',
    'manmade',
    'vegetation',
    'free',
]
# 红橙色系(车): 吊车红色、拖车洋红、卡车暗红、公交车粉红、汽车橙色
# 黄色系(小型交通参与者): 行人黄色、摩托车葱黄、自行车棕黄
# 绿色: 树
# 蓝色系(静态障碍): 围挡障碍蓝色、交通锥靛蓝
# 靛紫色系(路面): 路面靛色、人行道浅浅紫、地形浅紫
# 黑白色系(墙面隔断): 墙面浅灰、道中屏障(常见于双向车道高速路)深灰、通用障碍黑色
occ3d_colormap = [
    [0  , 0  , 0  ],  # 0  others               通用障碍     黑色
    [0  , 0  , 255],  # 1  barrier              围挡障碍     蓝色
    [174, 112, 0  ],  # 2  bicycle              自行车      棕黄
    [255, 180, 170],  # 3  bus                  公交车      粉红
    [255, 128, 0  ],  # 4  car                  汽车        橙色
    [255, 0  , 0  ],  # 5  construction_vehicle 建筑车辆吊车  红色
    [163, 217, 0  ],  # 6  motorcycle           摩托车      葱黄
    [255, 255, 0  ],  # 7  pedestrian           行人        黄色
    [6  , 82 , 121],  # 8  traffic_cone         交通锥      靛蓝
    [255, 0  , 150],  # 9  trailer              拖车        洋红
    [160, 0  , 0  ],  # 10 trunk                卡车        暗红
    [0  , 255, 255],  # 11 driveable_surface    路面        靛色
    [100, 100, 100],  # 12 other_flat           道中屏障     深灰
    [255, 180, 255],  # 13 sidewalk             人行道       浅浅紫
    [255, 100, 255],  # 14 terrain    地形即人行道外的路面     浅紫
    [200, 200, 200],  # 15 manmade         人造建筑如墙等     浅灰
    [0  , 180, 0  ],  # 16 vegetation           植被        绿色
]
