from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf

from utils.etl import ETL_2D
from utils.etl import ETL_Functions
from utils.etl import leave_first, ignore_key, ignore_tensor

from create_ml_dataset import Microstructure_Breaker_Upper

from utils import typings


def predict_and_rmse(
    model,
    dataset: tf.data.Dataset,
    images: np.ndarray
) -> Tuple[np.ndarray, tf.Tensor]:
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
    norm_metadata: typings.Metadata_Normalizations,
    circle_data: List[typings.Circle_Info],
    micro_data: Tuple[str, int, int],
    pore_encoding: int,
    padding_encoding: int,
    user_params: Tuple[int, int, int, int],
    arr_imgs: np.ndarray = None,
    arr_meta: List[typings.Metadata] = None,
) -> Tuple[tf.data.Dataset, np.ndarray, List[typings.Metadata]]:
    r''' micro_to_dataset_loader takes a 2D electrode microstructure with
    spherical electrode particles and creates a tf.data.Dataset loader amenable
    for predictions using a Neural Network.

    Since the image extraction process is computationally expensive, specifying
    the "arr_imgs" and "arr_meta" data after running it for the first time will
    save time. Thus subsequent calls to create datasets corresponding to
    different C-rates and time steps is much quicker.

    Inputs:
    - micro_data: (micro_fname: str, cell_length: int, cell_width: int)
    - user_params: (width_wrt_radius: int, scale: int, output_img_size: int,
                    order: int)
    '''
    _, _, img_size, order = user_params

    # This only runs for the first time. Extracting images is computationally
    # expensive.
    if arr_imgs is None or arr_meta is None:

        arr_imgs, arr_meta = _prep_micro_data(
            circle_data,
            micro_data,
            user_params,
            pore_encoding,
            padding_encoding,
        )

    arr_meta = _set_rate_and_time(
        c_rate,
        time,
        arr_meta,
    )

    hash_meta = _meta_arr_to_hash(
        arr_meta
    )

    process_input_fns = [
        ignore_tensor(leave_first(
            ETL_Functions.gather_img_from_np_arr,
            arr_imgs,
            img_size,
            tf_img_size,
            order,
        )),
        ignore_key(leave_first(
            tf.math.divide,
            # Normalize by the max value for np.uint16
            tf.cast(2 ** 16 - 1, dtype=tf.float32),
        )),
    ]
    process_target_fns = [
        ignore_tensor(leave_first(
            ETL_Functions.fake_output,
            tf_img_size,
        ))
    ]

    etl = ETL_2D(
        hash_meta,
        norm_metadata,
        [i for i in range(len(circle_data))],
        batch_size,
        process_input_fns,
        process_target_fns,
    )
    dataset_loader = etl.get_ml_dataset()

    return dataset_loader, arr_imgs, arr_meta


def _prep_micro_data(
    circle_data: List[typings.Circle_Info],
    micro_data: Tuple[str, int, int],
    user_params: Tuple[int, int, int, int],
    pore_encoding: int,
    padding_encoding: int,
) -> Tuple[np.ndarray, List[typings.Metadata]]:
    r''' Extracts each from an electrode microstructure, centering it with a
    constant "box radius" as well as preparing the metadata.
    '''
    micro_fname, cell_length, cell_height = micro_data
    width_wrt_radius, scale, output_img_size, order = user_params

    mbu = Microstructure_Breaker_Upper(
        "",
        "",
        np.load(micro_fname),
        cell_length,
        cell_height,
        circle_data,
        -1,
        pore_encoding,
        padding_encoding,
        width_wrt_radius,
        scale,
    )

    arr_imgs, arr_meta = mbu.extract_particles_from_microstructure(
        width_wrt_radius,
        output_img_size,
        order,
    )

    return arr_imgs, arr_meta


def _set_rate_and_time(
    c_rate: float,
    time: float,
    arr_meta: List[typings.Metadata]
) -> List[typings.Metadata]:

    for idx, hash in enumerate(arr_meta):
        hash["c_rate"] = str(c_rate)
        hash["time"] = str(time)

        arr_meta[idx] = hash

    return arr_meta


def _meta_arr_to_hash(
    arr_meta: List[typings.Metadata],
) -> Dict[str, typings.Metadata]:
    ret = {}
    for idx, hash in enumerate(arr_meta):
        ret[str(idx)] = hash
    return ret
