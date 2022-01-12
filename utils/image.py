from typing import Tuple

import numpy as np
from scipy.ndimage import zoom
from math import ceil


def extract_input(box_radius: int,
                  micro_im: np.array,
                  circle: Tuple[str, str, str],
                  scale: int) -> Tuple[np.array, np.array]:
    r'''
    Gets both the input (blank)

    Returns: (input_im)
    '''

    x, y, _ = circle

    padded_micro = pad_image(micro_im, box_radius)

    # New coordinates for (x, y) after padding
    x_new, y_new = padded_coords(
        x,
        y,
        box_radius,
        scale)

    # Extract the "blank" microstructure image and image with particle with
    # color
    input_im = padded_micro[
        y_new - box_radius: y_new + box_radius - 1,
        x_new - box_radius: x_new + box_radius - 1,
        :
    ]

    return input_im


def pad_image(orig_im: np.array,
              pad_width: int,
              pad_type="constant") -> np.array:
    ret_im = np.copy(orig_im)
    ret_im = np.pad(ret_im,
                    (
                        (pad_width, pad_width),
                        (pad_width, pad_width),
                        (0, 0),
                    ), pad_type)

    return ret_im


def padded_coords(
        x: str,
        y: str,
        pad_width: int,
        scale=10) -> Tuple[int, int]:

    y_new = ceil(float(y) * scale) + pad_width
    x_new = ceil(float(x) * scale) + pad_width

    return (x_new, y_new)


def zoom_image(
    img: np.array,
    output_img_size: int = 200
) -> Tuple[np.array, float]:

    img_size, _, _ = img.shape
    zoom_factor = output_img_size / img_size
    zoom_tuple = (zoom_factor,) * 2 + (1,)

    ret_im = zoom(img, zoom_tuple)
    return ret_im, zoom_factor
