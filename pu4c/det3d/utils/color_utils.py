# rgb 排列组合 27 种去除黑白，并把七色环颜色前移
# color_det = [[r*0.5, g*0.5, b*0.5] for r in range(2, -1, -1) for g in range(2, -1, -1) for b in range(2, -1, -1)]
color_det_class25 = [
    [0.0, 1.0, 0.0], # 绿
    [1.0, 0.0, 0.0], # 红
    [1.0, 0.5, 0.0], # 橙
    [1.0, 1.0, 0.0], # 黄
    [0.0, 0.0, 1.0], # 蓝
    [0.0, 1.0, 1.0], # 靛
    [1.0, 0.0, 1.0], # 紫
    [1.0, 1.0, 0.5],
    [1.0, 0.5, 1.0],
    [1.0, 0.5, 0.5],
    [1.0, 0.0, 0.5],
    [0.5, 1.0, 1.0],
    [0.5, 1.0, 0.5],
    [0.5, 1.0, 0.0],
    [0.5, 0.5, 1.0],
    [0.5, 0.5, 0.5],
    [0.5, 0.5, 0.0],
    [0.5, 0.0, 1.0],
    [0.5, 0.0, 0.5],
    [0.5, 0.0, 0.0],
    [0.0, 1.0, 0.5],
    [0.0, 0.5, 1.0],
    [0.0, 0.5, 0.5],
    [0.0, 0.5, 0.0],
    [0.0, 0.0, 0.5],
]

def plot_color_list(color_rgb_list, strip_height=10, width=100):
    from matplotlib import pyplot as plt
    import numpy as np

    height = len(color_rgb_list) * strip_height
    img_grey = np.zeros((height, width, 1), dtype=np.uint8)
    color_idx = 0
    for start in range(0, height, strip_height):
        img_grey[start:start+strip_height] = color_idx
        color_idx += 1
    img_rgb = np.array(color_rgb_list)[img_grey.reshape(-1)].reshape(height, width, 3)

    plt.imshow(img_rgb)
