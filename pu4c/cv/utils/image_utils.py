import numpy as np

def center_crop(data: np.ndarray, crop_width, crop_height):
    height, width, _ = data.shape
    start_x = (width - crop_width) // 2
    start_y = (height - crop_height) // 2
    cropped_image = data[start_y:start_y + crop_height, start_x:start_x + crop_width, :]
    return cropped_image
