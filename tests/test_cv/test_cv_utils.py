import pytest
import pu4c
import numpy as np
import matplotlib.pyplot as plt

@pytest.mark.skip(reason='visualization func')
def test_lidar_to_rangeview():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    points, boxes3d, boxes3d_with_label = datadb.get("kitti/000010/points")

    fov_up, fov_down, height, width = np.radians(2), np.radians(-24.8), 64, 720 # kitti velodyne params
    # range_image, point_idx = pu4c.cv.utils.range_projection(
    #     points[:, :3], height, width, fov=[fov_up, fov_down],
    # )
    # range_image, point_idx = pu4c.cv.utils.lidar_to_rangeview(
    #     points[:, :3], height, width, fov=[fov_up, fov_down, -np.pi, np.pi], 
    # )
    resolution, height, width = [np.radians(26.8/256), np.radians(360.0/1024)], 256, 1024 # 垂直/水平视场角宽度 除以 height/width
    # resolution, height, width = [np.radians(26.8/64), np.radians(360.0/4096)], 64, 4096
    range_image, point_idx, intensity_image = pu4c.cv.utils.lidar_to_rangeview(
        points[:, :4], height, width, resolution=resolution, fov_offset_down=np.radians(-24.8), return_intensity=True
    )

    valid_pixels = [[y, x] for y in range(height) for x in range(width) if range_image[y, x] != -1]
    y, x = valid_pixels[666]
    depth, distance = range_image[y, x], np.linalg.norm(points[point_idx[y, x], :3], ord=2, axis=0)
    print(depth, distance, (depth == distance))
    plt.imsave('work_dirs/lidar_to_rangeview.png', range_image)

    # points2 = pu4c.cv.utils.rangeview_to_lidar(range_image, fov=[fov_up, fov_down, -np.pi, np.pi])
    points2 = pu4c.cv.utils.rangeview_to_lidar(
        range_image, resolution=resolution, fov_offset_down=np.radians(-24.8), intensity_image=intensity_image
    )
    print(points.shape[0], points2.shape[0])

    point_labels = np.concatenate((np.zeros(points.shape[0]), np.ones(points2.shape[0])), axis=0)
    points = np.concatenate((points[:, :3], points2[:, :3]), axis=0)
    pu4c.cv.cloud_viewer(points=points, point_labels=point_labels, rpc=True)


if __name__ == '__main__':
    test_lidar_to_rangeview()
