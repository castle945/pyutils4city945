from . import config
from pu4c.common.utils import rpc_func

# 点云可视化
@rpc_func
def cloud_viewer(filepath=None, num_features=4, transmat=None,          # read_points
    points=None, point_labels=None, ds_voxel_size=None,
    cloud_uniform_color=None, cloud_colormap=None,                      # pointcloud color
    boxes3d=None, 
    vis=None, show_axis=True, run=True, 
    rpc=False):
    """
    快速查看单帧点云，支持 pcd/bin/npy/pkl/txt，输入文件路径或 ndarray 数组
    输入点云及带标签的边界框，可用于三维目标检测可视化
    输入点云及点云标签，可用于三维语义分割可视化、体素中心点及体素标签可视化、对比处理前后点云
    Examples:
        pu4c.det3d.app.cloud_viewer(filepath="/datasets/KITTI/object/training/velodyne/000000.bin", num_features=4)
        pu4c.det3d.app.cloud_viewer(points, boxes3d=boxes3d, rpc=True)
        pu4c.det3d.app.cloud_viewer(points=points, boxes3d=boxes3d_with_label, cloud_uniform_color=[0.99,0.99,0.99])
        pu4c.det3d.app.cloud_viewer(points=points, point_labels=point_labels, cloud_colormap=colormap)
    Keys:
        -/=: 调整点云点的大小
    Args:
        cloud_uniform_color: 自定义纯色点云颜色，例如白色 [1,1,1]
        cloud_colormap: 点云标签颜色表
        boxes3d: (N, 7)[xyz,lwh,yaw] or (N, 8)[xyz,lwh,yaw,cls]
        show_axis: 是否绘制坐标轴，如果不绘制那么会自动调整观察视角
        rpc: False 本地执行，True 远程执行
    """
    import open3d as o3d
    import numpy as np
    from .utils import common_utils, open3d_utils
    
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 1
        vis.get_render_option().background_color = np.zeros(3)
    if show_axis:
        axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_geometry)

    if filepath is not None:
        points = common_utils.read_points(filepath, num_features=num_features, transmat=transmat)
    if points is not None:
        cloud_geometry = open3d_utils.create_pointcloud_geometry(
            points, labels=point_labels, ds_voxel_size=ds_voxel_size, 
            uniform_color=cloud_uniform_color, colormap=cloud_colormap, 
            )
        vis.add_geometry(cloud_geometry)
    if boxes3d is not None:
        boxes3d_geometry = open3d_utils.create_boxes3d_geometry(boxes3d)
        vis.add_geometry(boxes3d_geometry)

    if run:
        vis.run()
        vis.destroy_window()
@rpc_func
def voxel_viewer(voxel_centers, voxel_size, voxel_labels=None, voxel_colormap=None, show_axis=True, rpc=False):
    """
    输入体素中心点，可用于体素可视化
    输入体素中心点及标签，可以于 OCC 可视化
    Examples:
        pu4c.det3d.app.voxel_viewer(voxel_centers=voxel_coords*voxel_size, voxel_size=voxel_size)
        pu4c.det3d.app.voxel_viewer(voxel_centers, voxel_size, voxel_labels=labels, voxel_colormap=colormap)
    """
    import open3d as o3d
    import numpy as np
    from .utils import open3d_utils
    
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 5
    vis.get_render_option().background_color = [1, 1, 1]
    if voxel_centers is not None:
        voxels_geometry = open3d_utils.create_voxels_geometry(voxel_centers, voxel_size)
        vis.add_geometry(voxels_geometry)

    # 如有标签优先按标签着色，否则着纯色
    cloud_viewer(points=voxel_centers, point_labels=voxel_labels, 
        cloud_uniform_color=[0, 1, 0], cloud_colormap=voxel_colormap, 
        vis=vis, show_axis=show_axis, run=True,
        )
@rpc_func
def cloud_viewer_panels(points_list=None, point_labels_list=None, boxes3d_list=None, 
    cloud_uniform_color=None, cloud_colormap=None,                                  # pointcloud color
    run=True, show_axis=True, offset=None, 
    rpc=False):
    """
    Examples:
        pu4c.det3d.app.cloud_viewer_panels(points_list=[points1, points2], boxes3d_list=[boxes3d1, boxes3d2], offset=[180, 0, 0])
    Args:
        offset: 面板之间的间隔，open3d 窗口坐标系，右前上
    """
    import open3d as o3d
    import numpy as np
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.zeros(3)
    if show_axis:
        axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_geometry)

    offset = np.array([100, 0, 0]) if offset is None else np.array(offset)
    for i in range(len(points_list)):
        points = points_list[i][:, :3] + (offset * i)
        point_labels = point_labels_list[i] if point_labels_list is not None else None
        boxes3d = boxes3d_list[i] if boxes3d_list is not None else None
        if boxes3d_list is not None:
            boxes3d[:, :3] += (offset * i)
        cloud_viewer(
            points=points, point_labels=point_labels, boxes3d=boxes3d,
            cloud_uniform_color=cloud_uniform_color, cloud_colormap=cloud_colormap,
            vis=vis, show_axis=False, run=False, 
            )

    vis.run()
    vis.destroy_window()
@rpc_func
def cloud_player(root=None, pattern="*", num_features=4, filepaths=None,
    points_list=None, boxes3d_list=None, 
    cloud_uniform_color=None, show_axis=True,
    start=0, step=10, 
    rpc=False):
    """
    点云播放器，支持播放点云目录与点云列表
    Examples:
        pu4c.det3d.app.cloud_player(root="/datasets/KITTI/object/training/velodyne/", num_features=4, pattern="*.bin")
        pu4c.det3d.app.cloud_player(filepaths=filepaths, num_features=5, boxes3d_list=boxes3d_list, cloud_uniform_color=[0.99, 0.99, 0.99], rpc=True)
    Keys:
        A/D: pre/next one frame
        W/S: pre/next step frame
    """
    from glob import glob
    from .utils import open3d_utils

    assert (root is not None) or (filepaths is not None) or (points_list is not None)
    if root is not None:
        filepaths = sorted(glob(f'{root}/{pattern}'))
    length = len(points_list) if filepaths is None else len(filepaths)

    def switch(vis, i):
        vis.clear_geometries()
        print_msg = f"frame {i}" if root is None else f"frame {i}: {filepaths[i]}"
        print(print_msg)
        cloud_viewer(
            filepath=None if filepaths is None else filepaths[i],
            points=None if points_list is None else points_list[i],
            boxes3d=None if boxes3d_list is None else boxes3d_list[i],
            num_features=num_features,
            cloud_uniform_color=cloud_uniform_color, 
            vis=vis, show_axis=show_axis, run=False, 
            )
        # vis.poll_events()
        vis.update_renderer()
    
    open3d_utils.playcloud(switch, length, start=start, step=step)


@rpc_func
def plot_tsne2d(features, labels, 
    x="x", y="y", title="T-SNE",
    rpc=False):
    """
    Args:
        features: (N, M), N 个归一化的样本，每个 M 维
        labels: (N, ) 聚类标签
    """
    from sklearn.manifold import TSNE
    import numpy as np
    import pandas as pd
    import seaborn
    import matplotlib.pyplot as plt

    if features.shape[1] > 2:
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(features)

    df = pd.DataFrame({'x': features[:, 0], 'y': features[:, 1], 'label': labels})
    seaborn.scatterplot(
        data=df, x=x, y=y, hue=df.label, 
        palette=seaborn.color_palette("hls", len(np.unique(labels))),
    ).set(title=title)
    plt.show()
@rpc_func
def plot_umap(features, labels, 
    x="x", y="y", title="T-SNE",
    rpc=False):
    # 与 t-SNE 相比，它在保持数据全局结构方面更加出色，但更慢
    # see https://umap-learn.readthedocs.io/en/latest/auto_examples/plot_mnist_example.html
    import umap # pip install umap-learn
    import numpy as np
    import pandas as pd
    import seaborn
    import matplotlib.pyplot as plt

    if features.shape[1] > 2:
        features = umap.UMAP(random_state=0).fit_transform(features)

    df = pd.DataFrame({'x': features[:, 0], 'y': features[:, 1], 'label': labels})
    seaborn.scatterplot(
        data=df, x=x, y=y, hue=df.label, 
        palette=seaborn.color_palette("hls", len(np.unique(labels))),
    ).set(title=title)
    plt.show()

def det3d_test_data(root=config.test_data_root):
    from pu4c.common.utils import TestDataDB
    return TestDataDB(dbname="det3d_test_data", root=root)
