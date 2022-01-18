from typing import Dict, Tuple, List
from tqdm import tqdm

from .io import load_json
from .numerics import get_coords_in_circle, get_inscribing_meshgrid
from .image import extract_input, zoom_image, create_circle_activation
from .metrics import measure_porosity

import numpy as np
from skimage import io
import tensorflow as tf

from math import ceil

# Custom Typings

_META_DICT = Dict[str, float]
_CIRC_INFO = List[Dict[str, str]]


def electrode_colormap_from_predictions(
    input_dataset: tf.data.Dataset,
    predicted_imgs: np.array,
    norm_metadata: Tuple[int, int, int, float, int, int],
    scale: int = 10,
    grid_size: int = 1000,
):
    r''' electrode_colormap_from_predictions takes an electrode (in the form of
    an tf.data.Dataset) its predicted color output at a certain C-rate and time
    step to "patch" all the particles back into whole electrode, thus returning
    a colormap output.

    This output could be compared to ground-truth values from COMSOL to
    investigate the performance of the Machine Learning model, or as a stand-
    alone tool to get colormaps quicker than the Direct Numerical Solution
    solution.

    Note: "input_dataset" and "predicted_imgs" should be in the same order,
        i.e., they should not be shuffled. Otherwise, colors will not be in the
        correct order.

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

    L, h_cell, R_norm, zoom_norm, _, _ = norm_metadata

    electrode = np.zeros(
        shape=(h_cell * scale, L * scale, 3),
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
            x = meta[0] * L
            y = meta[1] * h_cell
            R = meta[2] * R_norm
            zoom_factor = meta[3] * zoom_norm

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


def tf_circle_activation(
    tf_img_size: int,
    width_wrt_radius: float,
    particle_color: np.array,
    scale: int = 10,
) -> tf.Tensor:
    r'''
    Inputs:
    - particle_color: color specified should be integers, e.g. [0, 128, 0]
    '''
    img = create_circle_activation(
        width_wrt_radius,
        scale,
    )

    img = (img * particle_color) / 255.

    ret_im = tf.convert_to_tensor(img)
    ret_im = tf.cast(ret_im, tf.float32)
    ret_im = tf.image.resize(ret_im, (tf_img_size, tf_img_size))

    return ret_im


def predict_and_rmse(
    model,
    dataset: tf.data.Dataset,
    images: np.array
) -> Tuple[np.array, tf.Tensor]:
    '''
    predict_and_rmse returns the predicted images as well as the Root Mean
    Square Error (RMSE) of the given dataset. Should be used when user has
    the target images, otherwise, the model's predict method is sufficient.

    Batch size of input_data and label_data should be the same.

    Inputs
    -----
    model: tensorflow/keras model
    input_data: np.array of image (x, y, RGB)
    label_data: np.array of image (x, y, RGB)

    Returns
    -----
    pred_ims: array of predicted images by the model, same batch size as
        input_data
    rmse: RMSE value. Should have size (1,)
    '''
    pred_ims = model.predict(dataset)
    rmse = tf.reduce_mean(tf.square(pred_ims - images))
    rmse = tf.sqrt(rmse)
    return pred_ims, rmse


######################################################
# "BREAK" ELECTRODE IN ML-"LEARNABLE" IMAGES/DATASET #
######################################################
def micro_to_dataset_loader(
    c_rate: float,
    time: int,
    tf_img_size: int,
    batch_size: int,
    norm_metadata: Tuple[int, int, int, float, int, int],
    circle_data: Tuple[str, int, str],
    micro_data: Tuple[str, np.array, int],
    user_params: Tuple[int, int, int],
    arr_imgs: np.array = None,
    arr_meta: List[_META_DICT] = None,
) -> Tuple[tf.data.Dataset, np.array, List[_META_DICT]]:
    r''' micro_to_dataset_loader takes a 2D electrode microstructure with
    spherical electrode particles and creates a tf.data.Dataset loader amenable
    for predictions using a Neural Network.

    Since the image extraction process is computationally expensive, specifying
    the "arr_imgs" and "arr_meta" data after running it for the first time will
    save time. Thus subsequent calls to create datasets corresponding to
    different C-rates and time steps is much quicker.

    Inputs:
    - circle_data: (circle_data_fname: str, micro_key: int, circle_key: str)
    - micro_data: (micro_fname: str, pore_color: np.array, cell_length: int)
    - user_params: (width_wrt_radius: int, scale: int, output_img_size: int)
    '''

    # This only runs for the first time. Extracting images is computationally
    # expensive.
    if arr_imgs is None or arr_meta is None:

        arr_imgs, arr_meta = _prep_micro_data(
            circle_data,
            micro_data,
            user_params,
        )

    arr_meta = _set_rate_and_time(
        c_rate,
        time,
        arr_meta,
    )

    normed_arr_meta = _norm_metadata(
        arr_meta,
        norm_metadata,
    )

    dataset_loader = _micro_data_to_tf_loader(
        arr_imgs,
        normed_arr_meta,
        tf_img_size,
        batch_size,
    )

    return dataset_loader, arr_imgs, arr_meta


def _norm_metadata(
    arr_meta: List[str],
    norm_metadata: Tuple[int, int, int, float, int, int],
) -> List[str]:
    L, h_cell, R_max, zoom_norm, c_rate_norm, time_norm = norm_metadata

    ret = ["" for _ in range(len(arr_meta))]

    for idx, data in enumerate(arr_meta):
        x = float(data["x"]) / L
        y = float(data["y"]) / h_cell
        R = float(data["R"]) / R_max
        zoom = float(data["zoom_factor"]) / zoom_norm
        c_rate = float(data["c-rate"]) / c_rate_norm
        time = float(data["time"]) / time_norm
        porosity = float(data["porosity"])
        dist_from_sep = float(data["dist_from_sep"])

        as_float = [x, y, R, zoom, c_rate, time, porosity, dist_from_sep]

        s = "-"
        idx_str = s.join([str(num) for num in as_float])

        ret[idx] = idx_str

    return ret


def _micro_data_to_tf_loader(
    arr_imgs: np.array,
    arr_meta: List[str],
    tf_img_size: int,
    batch_size: int = 32,
):
    AUTOTUNE = tf.data.AUTOTUNE

    idx_nums = [idx for idx in range(len(arr_meta))]
    starting_ds = tf.data.Dataset.from_tensor_slices(idx_nums)

    def process_inputs(idx_num):
        input_im = _load_image(
            idx_num,
            arr_imgs,
            tf_img_size)

        metadata = _format_metadata(
            idx_num,
            arr_meta,
        )

        return input_im, metadata

    def fake_output(idx_num):
        blank_im = np.zeros((tf_img_size, tf_img_size, 3))
        blank_im = tf.convert_to_tensor(blank_im)
        return blank_im

    def configure_for_performance(ds):
        ds = ds.cache()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    inp_ds = starting_ds.map(process_inputs, num_parallel_calls=AUTOTUNE)
    inp_ds = configure_for_performance(inp_ds)

    out_ds = starting_ds.map(fake_output, num_parallel_calls=AUTOTUNE)
    ret_ds = tf.data.Dataset.zip((inp_ds, out_ds))

    return ret_ds


def _load_image(
    idx_num,
    arr_imgs: np.array,
    im_size: int,
):
    img = tf.gather(arr_imgs, idx_num)
    img = tf.convert_to_tensor(img)
    img = tf.image.resize(img, (im_size, im_size))
    return img


def _format_metadata(
    idx_num,
    arr_meta: List[str],
):
    as_str = tf.gather(arr_meta, idx_num)
    str_nums = tf.strings.split(as_str, sep="-")
    ret = tf.strings.to_number(str_nums, out_type=tf.dtypes.float32)
    return ret


def _prep_micro_data(
    circle_data: Tuple[str, int, str],
    micro_data: Tuple[str, np.array, int],
    user_params: Tuple[int, int, int],
) -> Tuple[np.array, List[_META_DICT]]:
    r''' Extracts each from an electrode microstructure, centering it with a
    constant "box radius" as well as preparing the metadata.
    '''

    circle_data_fname, micro_key, circle_key = circle_data
    micro_fname, pore_color, cell_length = micro_data
    width_wrt_radius, scale, output_img_size = user_params

    circle_data = _get_circ_data(
        circle_data_fname,
        micro_key,
        circle_key,
    )

    micro_im = io.imread(micro_fname)

    arr_imgs, arr_meta = _extract_image_and_meta(
        circle_data,
        micro_im,
        pore_color,
        cell_length,
        width_wrt_radius,
        scale,
        output_img_size,
    )

    return arr_imgs, arr_meta


def _set_rate_and_time(
    c_rate: float,
    time: float,
    arr_meta: List[_META_DICT]
) -> List[_META_DICT]:

    for idx, hash in enumerate(arr_meta):
        hash["c-rate"] = c_rate
        hash["time"] = time

        arr_meta[idx] = hash

    return arr_meta


def _extract_image_and_meta(
    circ_data: _CIRC_INFO,
    micro_im: np.array,
    pore_color: np.array,
    cell_length: int,
    width_wrt_radius: int = 3,
    scale: int = 10,
    output_img_size: int = 200,
) -> Tuple[np.array, List[_META_DICT]]:
    num_circs = len(circ_data)

    arr_imgs = np.zeros(
        (num_circs,
         output_img_size,
         output_img_size,
         3),
        dtype=np.uint8)
    arr_meta = []

    for idx, hash in tqdm(enumerate(circ_data)):
        x = hash['x']
        y = hash['y']
        r = hash['R']

        # Size image according to radius factor
        box_radius = ceil(float(r) * scale * width_wrt_radius)

        circ_im = extract_input(
            box_radius,
            micro_im,
            (x, y, r),
            scale
        )

        porosity = measure_porosity(
            circ_im,
            pore_color,
        )

        zoomed_circ_im, zoom_factor = zoom_image(
            circ_im,
            output_img_size,
        )

        circ_meta = {
            'x': x,
            'y': y,
            'R': r,
            'zoom_factor': zoom_factor,
            'porosity': porosity,
            'dist_from_sep': float(x) / cell_length,
        }

        arr_imgs[idx] = zoomed_circ_im
        arr_meta.append(circ_meta)

    return arr_imgs, arr_meta


def _get_circ_data(
    filename: str = "metadata.json",
    micro_key: int = 1,
    circle_key: str = "circles",
) -> _CIRC_INFO:
    microstructures = load_json(filename)
    circle_data = microstructures[micro_key - 1][circle_key]
    return circle_data


if __name__ == "__main__":
    input_imgs = micro_to_dataset_loader(
        4,
        225,
        99,
        32,
        (176, 100, 10, 4.918, 4, 10800),
        ("metadata.json", 1, "circles"),
        ("micro_1.png", np.array([0, 128, 0]), 176),
        (1, 10, 300),
    )
