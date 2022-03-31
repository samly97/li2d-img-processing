from typing import Tuple

from utils.numerics import get_coords_in_circle
from utils.numerics import get_inscribing_meshgrid
from utils.image import zoom_image

import numpy as np
import tensorflow as tf


def electrode_sol_map_from_predictions(
    input_dataset: tf.data.Dataset,
    predicted_imgs: np.ndarray,
    L_electrode: int,
    norm_metadata: Tuple[int, int, int, float, int, int],
    scale: int = 10,
    grid_size: int = 1000,
) -> np.ndarray:
    r''' electrode_sol_map_from_predictions takes an electrode (in the form of
    an tf.data.Dataset) its predicted SOL output at a certain C-rate and time
    step to "patch" all the particles back into whole electrode, thus returning
    a "State-of-Lithiation"-map output.

    This output could be compared to ground-truth values from COMSOL to
    investigate the performance of the Machine Learning model, or as a stand-
    alone tool to get SOL maps quicker than the Direct Numerical Solution
    solution.

    Note: "input_dataset" and "predicted_imgs" should be in the same order,
        i.e., they should not be shuffled. Otherwise, the SOL will not be in
        the correct order.

    Inputs:
    - input_dataset: tf.data.Dataset
    - predicted_imgs: np.array
    - L_electrode: electrode length (um)
    - norm_metadata: values used to normalize the metadata during model
        training
    - scale: int; scales the resolution of the outputted image.
    - grid_size: int; parameter used in constructing a meshgrid. A large enough
        value should be used fill in colors. Though a balance should be made
        for computational efficiency.
    '''

    _, h_cell, R_norm, zoom_norm, _, _ = norm_metadata

    electrode = np.zeros(
        shape=(h_cell * scale, L_electrode * scale, 1),
        dtype=float,
    )

    # For each input (image, metadata) in the dataset, extract the color from
    # the Machine Learning output and place it in the coordinate in the
    # electrode.
    img_idx = 0
    for batch in input_dataset:
        input_batch, _ = batch
        _, meta_array = input_batch

        for meta in meta_array:

            meta = meta.numpy()
            x = meta[0] * L_electrode
            y = meta[1] * h_cell
            R = meta[2] * R_norm
            zoom_factor = meta[4] * zoom_norm

            predicted_img = predicted_imgs[img_idx]
            img_size, _, _ = predicted_img.shape
            unzoomed_img, _ = zoom_image(
                predicted_img,
                img_size / zoom_factor,
            )

            unzoomed_img_size, _, _ = unzoomed_img.shape
            center = unzoomed_img_size / 2

            # The xx, yy coordinates could be shared between the electrode and
            # the ML output. The coordinates need to be translated to be in the
            # center of the image for resuse for the ML output images.
            xx, yy = get_inscribing_meshgrid(x, y, R, grid_size, to_um=1)
            xx, yy = get_coords_in_circle(x, y, R, (xx, yy), to_um=1)

            col_xx = np.copy(xx) - x
            col_yy = np.copy(yy) - y

            col_xx = np.floor(col_xx + center).astype(np.uint64)
            col_yy = np.floor(col_yy + center).astype(np.uint64)

            electrode_xx = np.ceil(xx * scale).astype(np.uint64) - 1
            electrode_yy = np.ceil(yy * scale).astype(np.uint64) - 1

            electrode[
                electrode_yy,
                electrode_xx,
                :,
            ] = unzoomed_img[
                col_yy,
                col_xx,
                :,
            ]

            img_idx += 1

    return electrode
