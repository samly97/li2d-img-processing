from utils.numerics import get_coords_in_circle
from utils.numerics import get_inscribing_meshgrid

from utils import typings

from scipy.interpolate import griddata

import numpy as np
import tensorflow as tf
import matplotlib


def electrode_mask_exceeding_absolute_SOL_error_threshold(
    predicted_sol: np.array,
    ground_truth_sol: np.array,
    electrode_mask: np.array,
    sol_threshold: float,
    over_predict: bool = True,
):
    blank_im = np.zeros_like(predicted_sol, dtype=np.float32)

    error = ground_truth_sol - predicted_sol

    if over_predict:
        error = error
    else:
        error = -error

    error = np.reshape(
        error,
        (np.prod(predicted_sol.shape), 1),
    )

    condition = np.all(
        error >= sol_threshold,
        axis=-1,
    )

    # Don't want to cause weird bugs do we
    exceed_error_mask = np.copy(electrode_mask)
    exceed_error_mask = np.reshape(
        exceed_error_mask,
        (np.prod(exceed_error_mask.shape), 1),
    )

    indices = np.where(condition)
    exceed_error_mask[:] = False
    exceed_error_mask[indices] = True

    exceed_error_mask = np.reshape(
        exceed_error_mask,
        blank_im.shape,
    )

    return exceed_error_mask


def electrode_colormap(
    sol_map: np.array,
    electrode_mask: np.array,
    colormap: matplotlib.cm,
    multiply_by_rgb: bool = True,
) -> np.array:
    r''' `electrode_colormap` takes in the predicted State-of-Lithiation values
    over the electrode and then returns a pretty colormap np.array ready to be
    saved as an image.

    Inputs:
    - sol_map: np.array | [h_cell, L, 1] | dtype = np.float32 [0, 1];
        State-of-Lithiation values over the electrode particles.
    - electrode_mask: np.array | [h_cell, L] | dtype = bool; from
        utils.image.electrode_mask_2D

    Outputs:
    - ret_im: np.array | [h_cell, L, 3] | dtype = np.uint8 [0, 255]; the
        colormap image in RGB channels. Ready to be saved as an image.
    '''

    Y, X, _ = sol_map.shape

    ret_im = np.zeros(
        (Y, X, 3), dtype=np.uint8,
    )

    sol = sol_map[electrode_mask]
    sol = np.reshape(sol, sol.size)

    rgb = colormap(sol)
    rgb = rgb[..., :3]
    if multiply_by_rgb:
        rgb = (rgb * 255).astype(np.uint8)

    ret_im[electrode_mask, :] = rgb

    return ret_im


def electrode_sol_map_from_predictions(
        input_dataset: tf.data.Dataset,
        predicted_imgs: np.ndarray,
        norm_metadata: typings.Metadata_Normalizations,
        width_wrt_radius: int,
        scale: int = 10,
        grid_size: int = 300,
):
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
    - norm_metadata: values used to normalize the metadata during model
        training
    - scale: int; scales the resolution of the outputted image.
    - grid_size: int; parameter used in constructing a meshgrid. A large enough
        value should be used fill in colors. Though a balance should be made
        for computational efficiency.
    '''
    L = norm_metadata["L"]
    h_cell = norm_metadata["h_cell"]
    R_norm = norm_metadata["R_max"]
    zoom_norm = norm_metadata["zoom_norm"]

    electrode = np.zeros(
        shape=(h_cell * scale, L * scale, 1),
        dtype=float,
    )

    img_idx = 0

    for batch in input_dataset:
        input_batch, _ = batch
        _, meta_batch = input_batch

        for meta in meta_batch:
            meta = meta.numpy()

            x_bigdaddy = meta[0] * L
            y_bigdaddy = meta[1] * h_cell
            R = meta[2] * R_norm
            zoom_factor = meta[3] * zoom_norm

            predicted_img = predicted_imgs[img_idx]
            img_size, _, _ = predicted_img.shape

            target_img_size = 1 / zoom_factor * img_size
            target_img_size = int(round(target_img_size))

            # Compute the center pixel for each case...
            c1 = img_size / 2
            c2 = target_img_size / 2

            # Radius on a pixel scale
            R1 = c1 / width_wrt_radius
            R2 = c2 / width_wrt_radius

            c1 = np.floor(c1).astype(np.uint32)
            c2 = np.floor(c2).astype(np.uint32)

            R1 = np.floor(R1).astype(np.uint32)
            R2 = np.floor(R2).astype(np.uint32)

            theta = np.linspace(0, 2*np.pi, 50)
            temp = np.linspace(1, R1, 50)
            r1 = np.ones((50, 50))
            r1 = r1 * temp
            r1 = r1.T

            x1 = c1 + r1 * np.cos(theta)
            y1 = c1 + r1 * np.sin(theta)

            x2 = c2 + r1 * R2/R1 * np.cos(theta)
            y2 = c2 + r1 * R2/R1 * np.sin(theta)

            x2 = np.ceil(x2).astype(np.uint32) - 1
            y2 = np.ceil(y2).astype(np.uint32) - 1

            dim = np.prod(np.array(x2.shape))

            x2 = np.reshape(x2, (dim,))
            y2 = np.reshape(y2, (dim,))

            x1 = np.ceil(x1).astype(np.uint32) - 1
            y1 = np.ceil(y1).astype(np.uint32) - 1

            x1 = np.reshape(x1, (dim,))
            y1 = np.reshape(y1, (dim,))

            holder = {}

            for idx in range(x1.shape[0]):
                x_new = int(x2[idx])
                y_new = int(y2[idx])

                new_coords = (y_new, x_new)

                y = int(y1[idx])
                x = int(x1[idx])

                interpolant = predicted_img[y, x, :]

                hash = holder.get(new_coords, {
                    "sum": 0,
                    "n": 0,
                })

                hash["sum"] += interpolant
                hash["n"] += 1

                holder[new_coords] = hash

            y_pic = np.zeros(len(holder))
            x_pic = np.zeros(len(holder))

            z = np.zeros(len(holder))

            for idx, coords in enumerate(holder):
                hash = holder[coords]
                s = hash["sum"]
                n = hash["n"]

                y_pic[idx] = coords[0]
                x_pic[idx] = coords[1]
                z[idx] = s/n

            x_lin = np.linspace(c2 - R2, c2 + R2, 300)
            y_lin = x_lin

            xx, yy = np.meshgrid(x_lin, y_lin)

            in_circ = (xx - c2) ** 2 + (yy - c2) ** 2 <= R2 ** 2

            xx = xx[in_circ] - 1
            yy = yy[in_circ] - 1

            points = np.array([x_pic, y_pic])
            points = points.T

            z_inter = griddata(
                points,
                z,
                (xx, yy),
                method="linear"
            )

            z_inter = np.reshape(z_inter, (z_inter.size, 1))

            xx = xx.astype(np.uint32)
            yy = yy.astype(np.uint32)

            blank_im = np.zeros((target_img_size, target_img_size, 1))
            blank_im[yy, xx, :] = z_inter

            unzoomed_img_size, _, _ = blank_im.shape
            center = unzoomed_img_size / 2

            # The xx, yy coordinates could be shared between the electrode and
            # the ML output. The coordinates need to be translated to be in the
            # center of the image for resuse for the ML output images.
            xx, yy = get_inscribing_meshgrid(
                x_bigdaddy, y_bigdaddy, R, grid_size, to_um=1)
            xx, yy = get_coords_in_circle(
                x_bigdaddy, y_bigdaddy, R, (xx, yy), to_um=1)

            col_xx = np.copy(xx) - x_bigdaddy
            col_yy = np.copy(yy) - y_bigdaddy

            col_xx = np.floor(col_xx + center).astype(np.uint64)
            col_yy = np.floor(col_yy + center).astype(np.uint64)

            electrode_xx = np.ceil(xx * scale).astype(np.uint64) - 1
            electrode_yy = np.ceil(yy * scale).astype(np.uint64) - 1

            electrode[
                electrode_yy,
                electrode_xx,
                :,
            ] = blank_im[
                col_yy,
                col_xx,
                :,
            ]

            img_idx += 1

    return electrode
