from pu4c.common.utils import rpc_func
import pu4c.config as cfg

# 点云可视化
@rpc_func
def cloud_viewer(
    filepath=None, num_features=4, transmat=None,          # read_points
    points=None, point_labels=None, ds_voxel_size=None,
    cloud_colormap=None, cloud_uniform_color=None,         # 点云着色，优先级 标签颜色映射 > 指定的纯色 > 反射率(points[:, 3])着色
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
        points: (N, 3)[x,y,z] or (N, 4)[x,y,z,i] 当 point_labels 为 None 时，如果 (N, 4) 则按反射率着色否则 open3d 默认按照高度着色
        boxes3d: (N, 7)[xyz,lwh,yaw] or (N, 8)[xyz,lwh,yaw,cls]
        ds_voxel_size: 降采样尺寸，注意如设置此值因为 open3d 降采样完之后只会保留坐标信息，将只能按高度对点云着色
        show_axis: 是否绘制坐标轴，如果不绘制那么会自动调整观察视角
        rpc: False 本地执行，True 远程执行
    """
    import open3d as o3d
    import numpy as np
    from .utils import read_points, open3d_utils
    
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 1
        vis.get_render_option().background_color = np.zeros(3)
    if show_axis:
        axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_geometry)

    if filepath is not None:
        points = read_points(filepath, num_features=num_features, transmat=transmat)
    if points is not None:
        cloud_geometry = open3d_utils.create_pointcloud_geometry(
            points, labels=point_labels, ds_voxel_size=ds_voxel_size, 
            colormap=cloud_colormap, uniform_color=cloud_uniform_color, 
            )
        vis.add_geometry(cloud_geometry)
    if boxes3d is not None:
        boxes3d_geometry = open3d_utils.create_boxes3d_geometry(boxes3d)
        vis.add_geometry(boxes3d_geometry)

    if run:
        vis.run()
        vis.destroy_window()
@rpc_func
def voxel_viewer(
    voxel_centers, voxel_size, 
    voxel_labels=None, voxel_colormap=None, 
    vis=None, run=True, 
    rpc=False):
    """
    输入体素中心点，可用于体素可视化
    输入体素中心点及标签，可以于 OCC 可视化
    Examples:
        pu4c.det3d.app.voxel_viewer(voxel_centers=voxel_coords*voxel_size, voxel_size=voxel_size)
        pu4c.det3d.app.voxel_viewer(voxel_centers, voxel_size, voxel_labels=labels, voxel_colormap=colormap)
    """
    import open3d as o3d
    from .utils import open3d_utils
    
    if vis is None:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 10
        vis.get_render_option().background_color = [1, 1, 1]
    if voxel_centers is not None:
        voxels_geometry = open3d_utils.create_voxels_geometry(voxel_centers, voxel_size)
        vis.add_geometry(voxels_geometry)

    # 如有标签优先按标签着色，否则着纯色
    cloud_viewer(points=voxel_centers, point_labels=voxel_labels, 
        cloud_uniform_color=[0, 1, 0], cloud_colormap=voxel_colormap, 
        vis=vis, show_axis=True, run=run,
        )
@rpc_func
def cloud_viewer_panels(points_list=None, point_labels_list=None, boxes3d_list=None, 
    cloud_uniform_color=None, cloud_colormap=None,                                  # pointcloud color
    show_axis=True, offset=None, 
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
    # @! 注意设置保存视角参数后，初始视角不再良好，需要滚动缩放一下才能看到
    Examples:
        pu4c.det3d.app.cloud_player(root="/datasets/KITTI/object/training/velodyne/", num_features=4, pattern="*.bin")
        pu4c.det3d.app.cloud_player(filepaths=filepaths, num_features=5, boxes3d_list=boxes3d_list, cloud_uniform_color=[0.99, 0.99, 0.99], rpc=True)
    Keys:
        A/D: pre/next one frame
        W/S: pre/next step frame
        Ctrl+C/V: 复制/粘贴视角参数
    """
    from glob import glob
    from .utils import open3d_utils

    assert (root is not None) or (filepaths is not None) or (points_list is not None)
    if root is not None:
        filepaths = sorted(glob(f'{root}/{pattern}'))
    length = len(points_list) if filepaths is None else len(filepaths)

    def switch(vis, i):
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
    
    open3d_utils.playcloud(switch, length, start=start, step=step)


# 图片可视化
@rpc_func
def image_viewer(filepath=None, data=None, rpc=False):
    """
    Args:
        data: (H, W, C) 图片数据
    """
    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    import numpy as np
    if data is None:
        assert filepath is not None
        data = imread(filepath)
    else:
        data = np.array(data, dtype=np.int32) # 只能可视化整型数据
    height, width = data.shape[:2]
    fig = plt.figure(figsize=(width, height), dpi=1)
    ax = fig.add_axes([0, 0, 1, 1]) # axes 是 figure 的内容，这里填充满 figure
    ax.imshow(data)
    ax.axis('off')
    plt.show() # cv2 可视化更方便但连续远程调用该函数会卡死


# 数据集可视化
@rpc_func
def play_semantickitti(
    root='/datasets/SemanticKITTI',
    start=0, step=10, 
    rpc=False):
    """播放 SemanticKITTI 数据集"""
    import glob, os
    import numpy as np
    from .utils import open3d_utils

    assert os.path.exists(root)
    sequencepaths = [os.path.join(root, 'dataset/sequences', str(i).zfill(2)) for i in range(11)] # 只有 seq0-10 是有标签的
    filepaths = sorted(glob.glob(f'{sequencepaths[0]}/labels/*.label'))
    global next_seq
    next_seq = 1
    colormap = np.array(cfg.semantickitti_colormap) / 255

    def switch(vis, i):
        if i >= len(filepaths):
            global next_seq
            print(f'loading frames from {sequencepaths[next_seq]}...')
            filepaths.extend(sorted(glob.glob(f'{sequencepaths[next_seq]}/labels/*.label')))
            next_seq += 1
        label_path = filepaths[i]
        print(f"frame {i}: {label_path}")
        labels = np.fromfile(label_path, dtype=np.uint32)
        labels = labels & 0xFFFF    # 等价于 % 2**16，取模后最大值为 255
        labels = np.vectorize(cfg.semantickitti_learning_map.__getitem__)(labels) # 映射完最大值 19
        # 如果要将 [0, 1-19] 的标签移动为 [0-18, 255] 则解注释，但那样未标记的点标签 255 不好写 colormap
        # ignore_label = 255          # 任意的一个大值，与有标记的语义类别区分开就行
        # labels[labels == 0] = ignore_label + 1  # 映射后的标签 0-19，其中 0 为未标记的点，将未标记的点赋值为 ignore_label，由于之后所有标签要 -1 故这里要赋值为 ignore_label+1
        # labels = labels - 1                     # 经上一行将未标记的点标签重新赋值后，此时 1-19 为有标记的点，将其标签整体 -1 则 0-18 为最终的语义标签，255 为未标记
        labels = labels.astype(np.uint8)
        cloud_viewer(
            filepath=label_path.replace('labels', 'velodyne')[:-5] + 'bin', num_features=4,
            point_labels=labels, cloud_colormap=colormap,
            vis=vis, show_axis=True, run=False, 
            )
    
    open3d_utils.playcloud(switch, length=18660, start=start, step=step)

@rpc_func
def play_occ3d_nuscenes(
    root='/datasets/nuScenes/Occ3D-nuScenes',
    start=0, step=10, 
    rpc=False):
    """
    播放 Occ3D-nuScenes 数据集
    Files:
        labels.npz: 整个文件类似于一个字典
            semantics: (H=80/0.4=200, W=80/0.4=200, D=6.4/0.4=16), 每个体素的语义标签
            mask_lidar: HWD, 激光雷达视角下的掩膜
            mask_camera: HWD, 相机视角下的掩膜
    """
    import glob, os
    import numpy as np
    from .utils import open3d_utils

    assert os.path.exists(root)
    # 如果数据集是以 nfs 挂载到本地计算机，一次性搜索所有文件太慢，动态加载
    # filepaths = sorted(glob.glob(f"{os.path.join(root, 'gts')}/scene-*/*/labels.npz"))
    scenepaths = sorted(glob.glob(f"{os.path.join(root, 'gts')}/scene-*"))
    filepaths = sorted(glob.glob(f'{scenepaths[0]}/*/labels.npz'))
    global next_scene
    next_scene = 1
    colormap = np.array(cfg.occ3d_colormap) / 255

    def switch(vis, i):
        if i >= len(filepaths):
            global next_scene
            print(f'loading frames from {scenepaths[next_scene]}...')
            filepaths.extend(sorted(glob.glob(f'{scenepaths[next_scene]}/*/labels.npz')))
            next_scene += 1
        label_path = filepaths[i]
        print(f"frame {i}: {label_path}")
        data = np.load(label_path)
        labels, lidar_mask, camera_mask = data['semantics'], data['mask_lidar'], data['mask_camera']
        voxel_size, point_cloud_range = [0.4, 0.4, 0.4], np.array([-40, -40, -1, 40, 40, 5.4])
        
        FREE_LABEL = 17                     # 空体素类别
        valid_mask = labels != FREE_LABEL   # 非空体素掩膜，HWD
        valid_idx = np.where(valid_mask)    # 非空体素下标，tuple(X, Y, Z), X.shape(N=非空体素个数,)
        labels = labels[valid_idx]          # 非空体素的标签，(N,)，值域 0-16 共 17 类
        voxel_centers = np.ascontiguousarray(np.array(valid_idx).T) * voxel_size + voxel_size * 0.5 # 非空体素坐标即中心点坐标，(N, 3)，等于下标乘以体素大小
        voxel_centers = voxel_centers - ((point_cloud_range[3:6] - point_cloud_range[0:3]) / 2) # 坐标原点移动到场景中心

        voxel_viewer(
            voxel_centers, voxel_size,
            voxel_labels=labels, voxel_colormap=colormap,
            vis=vis, run=False, 
            )
    
    open3d_utils.playcloud(switch, length=34149, start=start, step=step, point_size=10, background_color=[1, 1, 1])
@rpc_func
def play_surroundocc(
    root='/datasets/nuScenes/surroundocc',
    start=0, step=10, 
    rpc=False):
    """播放 surroundocc 数据集"""
    import glob, os
    import numpy as np
    from .utils import open3d_utils

    assert os.path.exists(root)
    filepaths = sorted(glob.glob(f"{os.path.join(root, 'samples')}/*.npy"))
    colormap = np.array(cfg.occ3d_colormap) / 255

    def switch(vis, i):
        label_path = filepaths[i]
        print(f"frame {i}: {label_path}")
        sparse_labels = np.load(label_path)
        voxel_size, point_cloud_range = [0.5, 0.5, 0.5], np.array([-50, -50, -5, 50, 50, 3])

        voxel_centers = np.vstack((sparse_labels[:, 0], sparse_labels[:, 1], sparse_labels[:, 2])).T * voxel_size # 非空体素下标乘以体素大小得到非空体素坐标
        voxel_centers = voxel_centers + point_cloud_range[0:3] # 坐标原点移动到场景中心
        labels = sparse_labels[:, 3]

        voxel_viewer(
            voxel_centers, voxel_size,
            voxel_labels=labels, voxel_colormap=colormap,
            vis=vis, run=False, 
            )
    
    open3d_utils.playcloud(switch, length=len(filepaths), start=start, step=step, point_size=10, background_color=[1, 1, 1])


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
