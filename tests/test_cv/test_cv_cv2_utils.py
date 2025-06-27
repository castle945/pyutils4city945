import pytest
try:
    import cv2
except ImportError:
    pytest.skip("cv2 is not installed, skipping all tests in this module", allow_module_level=True)

from pu4c.cv.utils.cv2_utils import photo_metric_distortion
import matplotlib.pyplot as plt
from matplotlib.image import imread
import copy
import numpy as np

def test_photo_metric_distortion():
    filepath = "/datasets/nuScenes/Fulldatasetv1.0/samples/CAM_BACK/n015-2018-07-18-11-07-57+0800__CAM_BACK__1531883536437525.jpg"
    data = imread(filepath)
    
    data1 = photo_metric_distortion(copy.deepcopy(data), brightness=100)
    data2 = photo_metric_distortion(copy.deepcopy(data), contrast=1.5)
    data3 = photo_metric_distortion(copy.deepcopy(data), saturation=3)
    data4 = photo_metric_distortion(copy.deepcopy(data), hue=180)
    plt.axis('off')
    plt.imsave('work_dirs/brightness_plus100.png', data1.astype(np.uint8))
    plt.imsave('work_dirs/contrast_mult1.5.png', data2.astype(np.uint8))
    plt.imsave('work_dirs/saturation_mult3.png', data3.astype(np.uint8))
    plt.imsave('work_dirs/hue_plus180.png', data4.astype(np.uint8))

if __name__ == '__main__':
    test_photo_metric_distortion()