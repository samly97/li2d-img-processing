from typing import Tuple

import numpy as np
from scipy.ndimage import zoom
from math import ceil

from utils.numerics import get_electrode_meshgrid


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


def create_circle_activation(
    width_wrt_radius: float,
    scale: int = 10,
) -> np.array:
    r''' create_circle_activation creates a white circle centered in a square
    image. The distance from the center of the circle to the edge is specified
    using "width_wrt_radius".
    '''

    _white = np.array([1., 1., 1., ])
    # These values were arbitrarily chosen
    _a_big_number = 500
    _R = 10.0

    x = int(_a_big_number / 2) * scale
    y = x

    micro_im = np.zeros(
        (_a_big_number * scale, _a_big_number * scale, 3),
        dtype=float,
    )

    xx, yy = get_electrode_meshgrid(_a_big_number, _a_big_number, scale)

    box_radius = ceil(_R * scale * width_wrt_radius)
    in_circ = np.sqrt((yy * scale - y) ** 2 +
                      (xx * scale - x) ** 2) <= _R * scale

    micro_im[in_circ] = _white

    ret_im = micro_im[
        y - box_radius: y + box_radius - 1,
        x - box_radius: x + box_radius - 1,
        :,
    ]

    return ret_im
