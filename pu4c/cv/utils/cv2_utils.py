import numpy as np
import cv2

def photo_metric_distortion(data, rgb=True, linear_stretch=False, brightness=None, contrast=None, saturation=None, hue=None):
    """
    图像处理之光度失真
    Args:
        linear_stretch: 是否对增强后的数据，值域线性拉伸到 [0, 255]
        brightness: 亮度绝对差值，正数调亮负数调暗
        contrast: 对比度尺度，像素值的倍率
    """
    data = np.array(data, dtype=np.float32)
    data_bgr = data[:, :, ::-1] if rgb else data
    if brightness is not None:
        data_bgr += brightness
    if contrast is not None:
        data_bgr *= contrast
    if saturation is not None or hue is not None:
        data_hsv = cv2.cvtColor(data_bgr, cv2.COLOR_BGR2HSV)
        if saturation is not None:
            data_hsv[..., 1] *= saturation
        if hue is not None:
            data_hsv[..., 0] += hue
            data_hsv[..., 0][data_hsv[..., 0] > 360] -= 360
            data_hsv[..., 0][data_hsv[..., 0] < 0] += 360
        data_bgr = cv2.cvtColor(data_hsv, cv2.COLOR_HSV2BGR)
    
    if linear_stretch:
        max_val, min_val = np.max(data_bgr), np.min(data_bgr)
        stretched_image = (data_bgr - min_val) / (max_val - min_val) * 255
        data_bgr = np.clip(stretched_image, 0, 255).astype(np.uint8)
    else:
        data_bgr = data_bgr.astype(np.int32)
    return data_bgr[:, :, ::-1] if rgb else data_bgr
