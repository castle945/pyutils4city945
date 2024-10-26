from pu4c.common.utils import rpc_func

# 点云可视化
@rpc_func
def cloud_viewer(filepath=None, num_features=4, transmat=None, ds_voxel_size=None, 
                 points=None, boxes3d=None, 
                 cloud_color_uniform=None, intensity_as_label=False,    # pointcloud color
                 boxes3d_color_uniform=[0, 1, 0],                       # boxes3d color
                 draw_axis=True, 
                 **kwargs):
    """
    快速查看单帧点云，支持 pcd/bin/npy/pkl/txt，输入文件路径或 ndarray 数组
    Examples:
        pu4c.det3d.app.cloud_viewer(filepath="/datasets/KITTI/object/training/velodyne/000000.bin", num_features=4)
        pu4c.det3d.app.cloud_viewer(points, boxes3d=boxes3d, rpc=True)
    Keys:
        -/=: 调整点云点的大小
    Args:
        boxes3d: (N, 7)[xyz,lwh,yaw] or (N, 8)[xyz,lwh,yaw,cls]
        cloud_color_uniform: 自定义纯色点云颜色，例如白色 [1,1,1]
        intensity_as_label: 点云反射强度字段 points[:, 3] 作为标签着色，常用于比较处理前后的同一点云
        draw_axis: 是否绘制坐标轴，如果不绘制那么会自动调整观察视角
        rpc: False 本地执行，True 远程执行
    """
    import open3d as o3d
    import numpy as np
    from .utils import common_utils, open3d_utils, color_utils
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.zeros(3)
    if draw_axis:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    if filepath is not None:
        points = common_utils.read_points(filepath, num_features=num_features, transmat=transmat)
    elif points is None:
        raise ValueError(f"filepath and points cannot both be None")

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points[:, :3])
    if ds_voxel_size is not None:
        cloud = cloud.voxel_down_sample(ds_voxel_size)
    if cloud_color_uniform is not None: cloud.paint_uniform_color(cloud_color_uniform)
    if intensity_as_label:
        cloud_color_map = [[255, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255]]
        colors = np.array(cloud_color_map)[points[:, 3].astype(np.int32)]
        cloud.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(cloud)

    if boxes3d is not None:
        open3d_utils.create_add_3d_boxes(boxes3d, vis=vis, color_map=color_utils.color_rings7_det)

    vis.run()
    vis.destroy_window()
@rpc_func
def cloud_viewer_from_dir(root, pattern="*", num_features=4, 
                          boxes3d=None, 
                          start=0, step=10, 
                          boxes3d_color_uniform=None, 
                          **kwargs):
    """
    播放点云目录，支持传入处理好的 boxes3d 数组
    Examples:
        pu4c.det3d.app.cloud_viewer_from_dir(root="/datasets/KITTI/object/training/velodyne/", num_features=4, pattern="*.bin")
    Keys:
        A/D: pre/next one frame
        W/S: pre/next step frame
    """
    from glob import glob
    from .utils import open3d_utils
    files = sorted(glob(f'{root}/{pattern}'))

    point_clouds = []
    for filepath in files:
        point_clouds.append({
            'filepath': filepath,
            'num_features': num_features,
        })
    
    open3d_utils.playcloud(point_clouds, 
                           boxes3d=boxes3d, 
                           start=start, step=step, uniform_color=boxes3d_color_uniform, 
                           )
