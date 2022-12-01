from typing import Tuple, List

import numpy as np
from scipy.ndimage import zoom
from math import ceil

from utils import typings
from utils.numerics import get_electrode_meshgrid


def extract_input(
    box_radius: int,
    micro_im: np.ndarray,
    circle: Tuple[str, str, str],
    scale: int,
    padding: int = 0
) -> np.ndarray:
    r'''
    Gets both the input (blank)

    Returns: (input_im)
    '''

    x, y, _ = circle

    padded_micro = pad_image(
        micro_im,
        box_radius,
        padding
    )

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


def pad_image(orig_im: np.ndarray,
              pad_width: int,
              padding_value: int = 0,
              pad_type="constant") -> np.ndarray:
    ret_im = np.copy(orig_im)
    ret_im = np.pad(
        ret_im,
        (
            (pad_width, pad_width),
            (pad_width, pad_width),
            (0, 0),
        ), pad_type,
        constant_values=(padding_value,),
    )

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
    img: np.ndarray,
    output_img_size: int = 200,
    order: int = 0,
) -> Tuple[np.ndarray, float]:

    img_size, _, _ = img.shape
    zoom_factor = output_img_size / img_size
    zoom_tuple = (zoom_factor,) * 2 + (1,)

    ret_im = zoom(img, zoom_tuple, order=order)
    return ret_im, zoom_factor


def electrode_mask_2D(
    circles: List[typings.Circle_Info],
    L: int,
    h_cell: int,
    R_factor: float = 1,
    scale: int = 10,
    mode: str = "circle",
) -> np.ndarray:
    r''' `electrode_mask_2D` returns a NumPy array where `True` values indicate
    where an electrode particle is. Circular particles are assumed.

    Inputs:
    - circles: list of dicts of {"x": , "y": , "R": } where (x, y, R) are in
        `um`.
    - L: length of the electrode `um`.
    - h_cell: height/width of electrode `um`.
    - scale: multiplicative factor to scale `electrode_mask` array.

    Outputs:
    - electrode_mask: np.array | dtype=bool
    '''

    electrode_mask = np.zeros(
        (h_cell * scale, L * scale),
        dtype=bool
    )

    x_lin = np.linspace(0, L * scale - 1, num=L * scale)
    y_lin = np.linspace(0, h_cell * scale - 1, num=h_cell * scale)

    xx, yy = np.meshgrid(x_lin, y_lin)

    for dict in circles:
        x = float(dict["x"])
        y = float(dict["y"])
        R = float(dict["R"])

        if mode == "circle":
            in_circ = np.sqrt((xx - x * scale) ** 2
                              + (yy - y * scale) ** 2) <= R * R_factor * scale
        elif mode == "ring":
            cond1 = np.sqrt((xx - x * scale) ** 2 + (yy - y *
                            scale) ** 2) <= R * scale
            cond2 = np.sqrt((xx - x * scale) ** 2
                            + (yy - y * scale) ** 2) >= R * scale * R_factor
            in_circ = cond1 == cond2

        electrode_mask[in_circ] = True

    return electrode_mask
