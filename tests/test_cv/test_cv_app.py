import pytest
import pu4c
import numpy as np

@pytest.mark.skip(reason='visualization func')
def test_cloud_viewer():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')

    # 也即三维目标检测可视化
    pu4c.cv.cloud_viewer(filepath="/datasets/KITTI/object/training/velodyne/000008.bin", num_features=4, rpc=True)
    points, _, _, boxes3d, labels = datadb.get("mmdet3d/kitti_000008")
    boxes3d_with_label = np.concatenate((boxes3d, labels[:, None]), axis=1)
    pu4c.cv.cloud_viewer(points=points, boxes3d=boxes3d_with_label, rpc=True)
    pu4c.cv.cloud_viewer(points=points, boxes3d=boxes3d, ds_voxel_size=0.5, rpc=True)
    pu4c.cv.cloud_viewer(points=points, boxes3d=boxes3d_with_label, cloud_uniform_color=[0.99,0.99,0.99], rpc=True)
    pu4c.cv.cloud_viewer_panels(points_list=[points, points], boxes3d_list=[boxes3d_with_label, boxes3d_with_label], offset=[180, 0, 0], rpc=True)

    # 也即三维语义分割可视化
    points, labels, classes, colormap = datadb.get("semantickitti/000000")
    pu4c.cv.cloud_viewer(points=points, point_labels=labels, cloud_colormap=colormap, rpc=True)

@pytest.mark.skip(reason='visualization func')
def test_voxel_viewer():
    # 也即三维占据预测可视化
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    voxel_centers, voxel_size, labels, classes, colormap = datadb.get("occ3d_nuscenes/scene0001-1e19d0")
    pu4c.cv.voxel_viewer(voxel_centers, voxel_size, voxel_labels=labels, voxel_colormap=colormap, rpc=True)

@pytest.mark.skip(reason='visualization func')
def test_image_viewer():
    filepath = "/datasets/nuScenes/Fulldatasetv1.0/samples/CAM_BACK/n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883536437525.jpg"
    pu4c.cv.image_viewer(filepath=filepath, rpc=True)

@pytest.mark.skip(reason='visualization func')
def test_tsne():
    try:
        import sklearn, umap, seaborn, pandas
    except:
        return
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    data, label = datadb.get('list2/sklearn/digits', None)
    pu4c.cv.plot_tsne2d(data, label, rpc=True)
    # pu4c.cv.plot_umap(data, label, rpc=True) # umap 版本变更暂未适配新版本


if __name__ == '__main__':
    test_cloud_viewer()
    # pu4c.cv.cloud_player(root='/datasets/KITTI/object/training/velodyne/', num_features=4, rpc=True)
    # pu4c.cv.play_occ3d_nuscenes()