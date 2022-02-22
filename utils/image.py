from typing import Tuple, List, Dict

import numpy as np
from scipy.ndimage import zoom
from math import ceil
import tensorflow as tf

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
    output_img_size: int = 200
) -> Tuple[np.ndarray, float]:

    img_size, _, _ = img.shape
    zoom_factor = output_img_size / img_size
    zoom_tuple = (zoom_factor,) * 2 + (1,)

    ret_im = zoom(img, zoom_tuple)
    return ret_im, zoom_factor


def create_circle_mask(
    width_wrt_radius: float,
    scale: int = 10,
) -> np.ndarray:
    r''' create_circle_mask creates a boolean circle centered in a square
    image. The distance from the center of the circle to the edge is specified
    using "width_wrt_radius".
    '''

    # These values were arbitrarily chosen
    _a_big_number = 500
    _R = 10.0

    x = int(_a_big_number / 2) * scale
    y = x

    micro_im = np.zeros(
        (_a_big_number * scale, _a_big_number * scale, 1),
        dtype=bool,
    )

    xx, yy = get_electrode_meshgrid(_a_big_number, _a_big_number, scale)

    box_radius = ceil(_R * scale * width_wrt_radius)
    in_circ = np.sqrt((yy * scale - y) ** 2 +
                      (xx * scale - x) ** 2) <= _R * scale

    micro_im[in_circ] = True

    ret_im = micro_im[
        y - box_radius: y + box_radius - 1,
        x - box_radius: x + box_radius - 1,
        :,
    ]

    return ret_im


def circle_mask_as_vector_of_indices(
    width_wrt_radius: float,
    img_size: int,
    scale: int = 10,
) -> tf.Tensor:
    r''' `circle_mask_as_vector_of_indices` takes a determines the location of
    a circular particle and determines the indices of where the particle is as
    a reshaped vector.

    Inputs:
    - width_wrt_radius: float; the distance from the center of the particle to
        the edge of the image. E.g., `width_wrt_radius = 3` fits 3 radii in
        that distance.
    - img_size: int; the image size fed to the Neural Network.
    - scale: int; magnifying factor (scale * 1 um) for higher resolution images

    Outputs:
    - mask: the indices of where the particle is if an [img_size, img_size]
        image was reshaped into a vector. `mask` is a `tf.Tensor` of
        `dtype=tf.int32`.
    '''

    mask = create_circle_mask(
        width_wrt_radius,
        scale=scale,
    )
    mask = tf.cast(mask, dtype=tf.int32)
    mask = tf.image.resize(mask, (img_size, img_size))

    mask = mask.numpy()
    # Check location of particle
    mask = np.all(mask == [1], axis=-1)

    # Reshape into a vector
    mask = np.reshape(mask, img_size * img_size)

    # Indices (or location) of particle
    mask = np.where(mask)
    mask = np.array(mask)
    mask = np.reshape(mask, (mask.size))

    mask = tf.cast(
        mask,
        dtype=tf.int32
    )

    return mask


def tf_circle_mask(
    tf_img_size: int,
    width_wrt_radius: float,
    particle_color: np.ndarray,
    scale: int = 10,
) -> tf.Tensor:
    r'''
    Inputs:
    - particle_color: color specified should be integers, e.g. [0, 128, 0]
    '''

    # Gets mask where `True` represents where the particle lives
    mask = create_circle_mask(
        width_wrt_radius,
        scale,
    )

    Y, X, _ = mask.shape
    mask = np.reshape(mask, (Y * X * 1))

    blank_img = np.zeros(
        (Y, X, 3),
        dtype=np.float32
    )
    blank_img = np.reshape(blank_img, (Y * X, 3))
    blank_img[mask, :] = particle_color / 255.
    blank_img = np.reshape(blank_img, (Y, X, 3))

    ret_im = tf.convert_to_tensor(blank_img)
    ret_im = tf.cast(ret_im, tf.float32)
    ret_im = tf.image.resize(ret_im, (tf_img_size, tf_img_size))

    return ret_im


def electrode_mask_2D(
    circles: List[Dict[str, str]],
    L: int,
    h_cell: int,
    scale: int = 10,
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

        in_circ = np.sqrt((xx - x * scale) ** 2
                          + (yy - y * scale) ** 2) <= R * scale

        electrode_mask[in_circ] = True

    return electrode_mask
