import numpy as np

# 投影
def project_points_to_pixels(points, image_shape, transform_mat):
    """
    y = Rx 即 y(4,N) = transform_mat @ (4, N) 即 y(N,4) = (N,4) @ transform_mat.T
    """
    points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1), dtype=np.float32))) # [N, 4]
    points_cam = (points_hom @ transform_mat.T)[:, :3]
    
    pixels_depth = points_cam[:, 2]
    pixels = (points_cam[:, :2].T / points_cam[:, 2]).T # (N, 2)[col, row]

    # remove points outside the image
    mask = pixels_depth > 0
    mask = np.logical_and(mask, pixels[:, 0] > 0)
    mask = np.logical_and(mask, pixels[:, 0] < image_shape[1])
    mask = np.logical_and(mask, pixels[:, 1] > 0)
    mask = np.logical_and(mask, pixels[:, 1] < image_shape[0])

    return pixels, pixels_depth, mask
def project_pixels_to_points(pixels, depth, transform_mat):
    """
    Args:
        pixels: (N, 2)[x,y]
        depth: (N,)
    """
    N = depth.shape[0]
    points_cam = np.zeros((N, 3))
    points_cam[:, 2] = depth
    points_cam[:, :2] = pixels * depth[:, np.newaxis].repeat(2, axis=1)

    points_cam_hom = np.hstack((points_cam, np.ones((N, 1), dtype=np.float32))) # [N, 4]
    points_hom = points_cam_hom @ transform_mat.T
    return points_hom[:, :3]

def range_projection(points, fov_up=np.radians(3), fov_down=np.radians(-25), height=64, width=720):
    """lidar_to_rangeview 的仅线性拉伸简化版"""
    fov = abs(fov_up) + abs(fov_down)
    depth = np.linalg.norm(points[:, :3], ord=2, axis=1) # 按行求二范数，即距离

    yaw, pitch = -np.arctan2(points[:, 1], points[:, 0]), np.arcsin(points[:, 2] / depth)
    proj_x = 0.5 * (yaw / np.pi + 1.0)            # yaw=[-pi, pi] to [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # pitch=[fov_up, fov_down] to [0.0, 1.0]
    proj_x *= width     # to [0.0, W]
    proj_y *= height    # to [0.0, H]

    # 坐标取整作为像素坐标
    proj_x = np.minimum(width - 1, np.floor(proj_x))
    proj_x = np.maximum(0, proj_x).astype(np.int32)  # to [0, W-1]
    proj_y = np.minimum(height - 1, np.floor(proj_y))
    proj_y = np.maximum(0, proj_y).astype(np.int32)  # to [0, H-1]

    range_image = np.full((height, width), -1, dtype=np.float32)  # [H,W] range (-1 is no data)
    point_idx = np.full((height, width), -1, dtype=np.int32)  # [H,W] index (-1 is no data)
    range_image[proj_y, proj_x] = depth
    point_idx[proj_y, proj_x] = np.arange(depth.shape[0])

    return range_image, point_idx
def lidar_to_rangeview(points, height=64, width=1024, 
                       fov_vertical=[3, -25], fov_horizon=[-180, 180],
                       resolution=None, fov_offset_down=None,
                       max_depth=None, return_intensity=False):
    """
    一般算法都是以线性拉伸的方式转 RV 图像，其实这就是一种柱坐标系体素栅格
    Examples:
        线性拉伸方式: lidar_to_rangeview(points, height=height, width=width, fov_vertical=fov_vertical, fov_horizon=fov_horizon)
        非线性拉伸方式: lidar_to_rangeview(points, height=height, width=width, resolution=resolution, fov_down=fov_down)
    Args:
        points: 点云，必须遵循前左上坐标系
        fov_*,height,width: 传感器参数
        resolution: None 则以线性拉伸的方式，否则传入角度的分辨率 [res_y, res_x]，按实际填入（类似给定 voxel_size 的体素化）
        fov_offset_down: 对底部 fov 的偏移量以使得画面整体居中
        return_intensity: 是否返回反射强度图像
    Returns: 
        range_image: 浮点数深度值，-1 表示无效点
        point_idx: 像素坐标到原始点云点坐标的索引，point = points[point_idx[y, x]] 或 point_idx.reshape(-1) rv_points = points[point_idx[point_idx != -1]]
        intensity_image: 反射强度，-1 表示无效点
    """
    depth = np.linalg.norm(points[:, :3], ord=2, axis=1) # 按行求二范数，即距离
    # 前左上坐标系，yaw 加负号反向一下以遵循从左到右递增的习惯
    yaw, pitch = -np.arctan2(points[:, 1], points[:, 0]), np.arcsin(points[:, 2] / depth)

    if resolution is None:
        fov_up, fov_down = np.radians(fov_vertical)
        fov_left, fov_right = np.radians(fov_horizon)
        fov_height = abs(fov_up) + abs(fov_down)
        fov_width = abs(fov_left) + abs(fov_right)
        
        proj_x = yaw * width / fov_width + width / 2.0    # fov_width 线性拉伸到 width，并将 x 轴原点移动到图像中心列以符合前左上坐标系
        proj_y = -((pitch + abs(fov_down)) * height / fov_height) + height # + abs(fov_down) 将 patch 范围移动到 [0, +]

        proj_x = np.minimum(width - 1, np.floor(proj_x))
        proj_x = np.maximum(0, proj_x).astype(np.int32)  # to [0, W-1]
        proj_y = np.minimum(height - 1, np.floor(proj_y))
        proj_y = np.maximum(0, proj_y).astype(np.int32)  # to [0, H-1]
    else:
        assert (fov_offset_down is not None)
        res_y, res_x = np.radians(resolution)
        fov_offset_down = np.radians(fov_offset_down)
        proj_x = yaw / res_x + width / 2.0
        proj_y = -((pitch + abs(fov_offset_down)) / res_y) + height

        mask_in_x = np.logical_and(np.floor(proj_x) >=0, np.floor(proj_x) < width)
        mask_in_y = np.logical_and(np.floor(proj_y) >=0, np.floor(proj_y) < height)
        mask_in = np.logical_and(mask_in_x, mask_in_y)
        proj_x, proj_y = proj_x[mask_in].astype(np.int32), proj_y[mask_in].astype(np.int32)
        depth, points = depth[mask_in], points[mask_in]

    range_image = np.full((height, width), -1, dtype=np.float32)  # [H,W] range (-1 is no data)
    point_idx = np.full((height, width), -1, dtype=np.int32)  # [H,W] index (-1 is no data)
    intensity_image = np.full((height, width), -1, dtype=np.float32) if return_intensity else None
    # 按深度降序，则之后对投影到同一个像素点的多个激光点会自然取距离最近的点
    indices = np.argsort(depth)[::-1]
    depth, proj_x, proj_y, points = depth[indices], proj_x[indices], proj_y[indices], points[indices]
    if max_depth is not None:
        depth[depth > max_depth] = -1

    range_image[proj_y, proj_x] = depth
    point_idx[proj_y, proj_x] = np.arange(depth.shape[0])
    if return_intensity:
        intensity_image[proj_y, proj_x] = points[:, 3]

    return range_image, point_idx, intensity_image
def rangeview_to_lidar(range_image, intensity_image=None, 
                       fov_vertical=[3, -25], fov_horizon=[-180, 180],
                       resolution=None, fov_offset_down=None):
    """
    modified from https://github.com/city945/LiDAR4D/blob/main/utils/convert.py
    Returns:
        points: (N,4) or (N,3)
    """
    height, width = range_image.shape
    proj_x, proj_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32), indexing="xy")

    if resolution is None:
        fov_up, fov_down = np.radians(fov_vertical)
        fov_left, fov_right = np.radians(fov_horizon)
        fov_height = abs(fov_up) + abs(fov_down)
        fov_width = abs(fov_left) + abs(fov_right)

        yaw = -((proj_x - width/2.0)*fov_width / width)
        pitch = (fov_up - proj_y*fov_height / height)
    else:
        assert (fov_offset_down is not None)
        res_y, res_x = np.radians(resolution)
        fov_offset_down = np.radians(fov_offset_down)
        yaw = -((proj_x - width/2.0) * res_x)
        pitch = ((height - proj_y) * res_y) - abs(fov_offset_down)

    dirs = np.stack([np.cos(pitch)*np.cos(yaw), np.cos(pitch)*np.sin(yaw), np.sin(pitch)], axis=-1)
    points = dirs * range_image.reshape(height, width, 1)
    if intensity_image is not None:
        points = np.concatenate([points, intensity_image.reshape(height, width, 1)], axis=2)

    num_features = 3 if intensity_image is None else 4
    points = points[np.where(range_image != -1)].reshape(-1, num_features)

    return points


# 采样
def farthest_point_sample(point, npoint):
    """
    Args:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)  # N个点中随机取1个点作为初始采样点
    for i in range(npoint):
        centroids[i] = farthest         # 新增一个采样点，迭代 npoint 次
        centroid = xyz[farthest, :]     # 以采样点为中心点，计算到其他点的距离之和
        dist = np.sum((xyz - centroid) ** 2, -1) # axis=-1按最高维度雷达，二维则按列加，即行和即x**2+y**2+z**2
        mask = dist < distance          # 偏离太远的点置 false 丢弃
        distance[mask] = dist[mask]     # 只为为 true 的点更新有效距离， python 允许以bool数组做下标掩膜
        farthest = np.argmax(distance, -1)  # 取离当前点有效距离最远的点作为下一个采样点
    point = point[centroids.astype(np.int32)]
    return point
