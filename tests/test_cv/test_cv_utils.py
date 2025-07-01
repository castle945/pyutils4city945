import pytest
import pu4c
import numpy as np
import matplotlib.pyplot as plt
import copy

VISUALIZE = False # 执行测试时关闭，调试时可打开
try:
    import cv2
    NO_CV2 = False
except ImportError:
    NO_CV2 = True

def test_lidar_to_rangeview():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data/')
    points, image, calib, boxes3d, labels = datadb.get('mmdet3d/kitti_000008')

    # HW=64*720 常规的投影，由于点云个数(10 万级别)远大于 HW，大量点投影到同一个像素而丢失
    fov_up, fov_down, height, width = np.radians(2), np.radians(-24.8), 64, 720 # kitti velodyne params
    range_image, point_idx = pu4c.cv.utils.range_projection(
        points[:, :3], height, width, fov=[fov_up, fov_down],
    )
    # range_image, point_idx = pu4c.cv.utils.lidar_to_rangeview(
    #     points[:, :3], height, width, fov=[fov_up, fov_down, -np.pi, np.pi], 
    # )
    # points2 = pu4c.cv.utils.rangeview_to_lidar(range_image, fov=[fov_up, fov_down, -np.pi, np.pi])
    plt.imsave('work_dirs/range_projection.png', range_image)

    # HW=64*4096 以线性拉伸的方式投影，HW 达到 10 万级，仅有少量点丢失
    # 这也说明转 RV 的过程就类似于做体素化，而且此体素化是受透视遮挡影响的，即 RV 图像就是一种柱坐标系体素栅格
    # resolution, height, width = [np.radians(26.8/256), np.radians(360.0/1024)], 256, 1024 # 垂直/水平视场角宽度 除以 height/width
    resolution, height, width = [np.radians(26.8/64), np.radians(360.0/4096)], 64, 4096
    range_image, point_idx, intensity_image = pu4c.cv.utils.lidar_to_rangeview(
        points[:, :4], height, width, resolution=resolution, fov_offset_down=np.radians(-24.8), return_intensity=True
    )
    plt.imsave('work_dirs/lidar_to_rangeview.png', range_image)
    
    # 查找 RV 像素点对应的点云点
    valid_pixels = [[y, x] for y in range(height) for x in range(width) if range_image[y, x] != -1]
    y, x = valid_pixels[666]
    depth, distance = range_image[y, x], np.linalg.norm(points[point_idx[y, x], :3], ord=2, axis=0)
    print(depth, distance, (depth == distance))

    points2 = pu4c.cv.utils.rangeview_to_lidar(
        range_image, resolution=resolution, fov_offset_down=np.radians(-24.8), intensity_image=intensity_image
    )
    print(points.shape[0], points2.shape[0])

    if VISUALIZE:
        # 对比查看原始点云和(转 RV 再转回的点云)，常规投影方式大量点丢失，线性拉伸方式只有少量点没对应上丢失
        point_labels = np.concatenate((np.zeros(points.shape[0]), np.ones(points2.shape[0])), axis=0)
        points = np.concatenate((points[:, :3], points2[:, :3]), axis=0)
        pu4c.cv.cloud_viewer(points=points, point_labels=point_labels, rpc=True)

@pytest.mark.skipif(NO_CV2, reason='cv2 is not installed, skipping this test')
def test_project_points_to_pixels():
    datadb = pu4c.common.utils.TestDataDB(dbname='pu4c_unittest_data', root='tests/data')
    points, image, calib, boxes3d, labels = datadb.get('mmdet3d/kitti_000008')
    image = (image * 255).astype(np.uint8) # matplotlib 读图后会归一化
    image_copy = copy.deepcopy(image)
    # 边界框投影到图像
    corners_array = np.array([pu4c.cv.utils.get_oriented_bounding_box_corners(b[:3], b[3:6], np.array([0, 0, b[6]])) for b in boxes3d])
    lines = pu4c.cv.utils.get_oriented_bounding_box_lines()
    corners_pixels, _, mask = pu4c.cv.utils.project_points_to_pixels(corners_array.reshape(-1, 3), image.shape, transform_mat=calib['lidar2img'])
    corners_pixels, mask = corners_pixels.reshape(-1, 8, 2), mask.reshape(-1, 8)
    mask = [all(box_mask) for box_mask in mask]
    corners_pixels_array = corners_pixels[mask]
    for pixels in corners_pixels_array:
        for line in lines:
            x0, y0 = int(pixels[line[0]][0]), int(pixels[line[0]][1])
            x1, y1 = int(pixels[line[1]][0]), int(pixels[line[1]][1])
            cv2.line(image, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
    plt.imsave('work_dirs/image_with_boxes3d.png', image)

    # 点云投影到图像
    image = copy.deepcopy(image_copy)
    pixels, pixels_depth, mask = pu4c.cv.utils.project_points_to_pixels(points, image_shape=image.shape, transform_mat=calib['lidar2img'])
    for x, y in pixels[mask]: # 图像坐标系，x-col 向右 y-row 向下，负数厚度表示绘制实心圆
        cv2.circle(image, center=(int(x), int(y)), radius=1, color=(255, 0, 0), thickness=-1)
    plt.imsave('work_dirs/image_with_points.png', image)
    
    # 像素投回点
    if VISUALIZE:
        mask = np.logical_and(mask, pixels_depth < 80)
        points = pu4c.cv.utils.project_pixels_to_points(pixels=pixels[mask], depth=pixels_depth[mask], transform_mat=np.linalg.inv(calib['lidar2img']))
        pu4c.cv.cloud_viewer(points=points, rpc=True)

if __name__ == '__main__':
    # test_lidar_to_rangeview()
    test_project_points_to_pixels()
