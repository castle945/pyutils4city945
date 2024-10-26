import open3d as o3d
import numpy as np
from pu4c.det3d.utils import common_utils, color_utils

def translate_boxes_to_open3d_instance(xyz, lwh, rpy):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(rpy)
    box3d = o3d.geometry.OrientedBoundingBox(xyz, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d
def create_add_3d_boxes(boxes3d, vis=None, color_map=None):
    """
    Args:
        boxes3d: (N, 7)[xyz,lwh,yaw] or (N, 8)[xyz,lwh,yaw,cls]
    """
    with_label = False if boxes3d.shape[1] == 7 else True
    geometries = []
    for box in boxes3d:
        line_set, box3d = translate_boxes_to_open3d_instance(box[:3], box[3:6], np.array([0, 0, box[6] + 1e-10]))
        line_set.paint_uniform_color(
            color_map[int(box[7])] if with_label else [0, 1, 0],
        )
        geometries.append(line_set)
        if vis is not None: vis.add_geometry(line_set) # vis.add_geometry(box3d) # 立方体
    return geometries

def playcloud(point_clouds, 
              boxes3d=None, 
              start=0, step=10, uniform_color=None, 
              ):
    def switch(vis, i):
        pc = point_clouds[i]
        print(f"frame {i}: {pc['filepath']}")
        vis.clear_geometries()

        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

        points = common_utils.read_points(
            pc['filepath'], num_features=pc['num_features'],
            transmat=pc['transmat'] if pc.get('transmat', None) is not None else None,
            )
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points[:, :3])
        if uniform_color is not None: cloud.paint_uniform_color(uniform_color)
        vis.add_geometry(cloud) # 离谱 update 没用，add 反而有效

        if boxes3d is not None and boxes3d[i] is not None:
            create_add_3d_boxes(boxes3d[i], vis, color_map=color_utils.color_rings7_det)

        # vis.poll_events()
        vis.update_renderer()

    def prev(vis):
        global g_idx
        g_idx = max(g_idx - 1, 0)
        switch(vis, g_idx)
    def next(vis):
        global g_idx
        g_idx = min(g_idx + 1, len(point_clouds)-1)
        switch(vis, g_idx)
    def prev_n(vis):
        global g_idx
        g_idx = max(g_idx - step, 0)
        switch(vis, g_idx)
    def next_n(vis):
        global g_idx
        g_idx = min(g_idx + step, len(point_clouds)-1)
        switch(vis, g_idx)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    vis.get_render_option().point_size = 1
    vis.get_render_option().background_color = np.zeros(3)

    vis.register_key_callback(ord('W'), prev_n)
    vis.register_key_callback(ord('S'), next_n)
    vis.register_key_callback(ord('A'), prev)
    vis.register_key_callback(ord('D'), next) # 按小写，但这里要填大写
    # vis.register_key_callback(ord(' '), next) # space

    global g_idx
    g_idx = start
    switch(vis, start)
    vis.run()
    vis.destroy_window()