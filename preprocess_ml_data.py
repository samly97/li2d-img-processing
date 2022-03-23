import os
from typing import Dict, List, Tuple

import numpy as np
from random import shuffle
from math import ceil
import tensorflow as tf

from utils.io import load_json
from utils.etl import ETL_Functions, ETL_2D
from utils.etl import leave_first, ignore_key, ignore_tensor

from utils import typings


def parse_raw_data(
    data_dir: str,
    input_dir: str,
) -> List[int]:

    data_dir = os.path.join(os.getcwd(), data_dir)
    input_im_dir = os.path.join(data_dir, input_dir)
    input_im_fname = np.array([fname for fname in os.listdir(input_im_dir)])

    pic_num = [int(fname.split(".npy")[0]) for fname in input_im_fname]

    return pic_num


def shuffle_dataset(
    pic_num: List[int],
) -> List[int]:

    # Shuffle list
    list_idx = list(range(len(pic_num)))
    shuffle(list_idx)

    # Shuffle particle number list
    pic_num_arr = np.array(pic_num)
    pic_num_arr = pic_num_arr[list_idx]

    return pic_num_arr.tolist()


def get_split_indices(
    trn_split: int,
    val_split: int,
    pic_num: List[int],
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:

    trn_idx = ceil(len(pic_num) * trn_split)
    val_idx = ceil(len(pic_num) * val_split) + trn_idx
    test_idx = len(pic_num)

    return ((0, trn_idx), (trn_idx, val_idx), (val_idx, test_idx))


def preprocess_ml_data() -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.data.Dataset]:
    r''' Define alternate entry point to when the script is not being called
    explicitly, i.e., cases other than "__name__ == __main__".
    '''
    settings = load_json("pipeline.json")

    norm_metadata: typings.Metadata_Normalizations = {
        "L": settings["L"],
        "h_cell": settings["h_cell"],
        "R_max": settings["R_max"],
        "zoom_norm": settings["zoom_max"],
        "c_rate_norm": settings["c-rate"],
        "time_norm": settings["time"],
        "local_tortuosity_norm": settings["tortuosity_max"],
    }

    dataset_json: Dict[str, typings.Metadata] = load_json(
        settings["dataset_data"],
        path=os.path.join(os.getcwd(), settings["data_dir"]),
    )

    pic_num = parse_raw_data(
        settings["data_dir"],
        settings["input_dir"]
    )
    pic_num = shuffle_dataset(pic_num)

    trn_idx, val_idx, test_idx = get_split_indices(
        settings["trn_split"],
        settings["val_split"],
        pic_num,
    )

    datasets = [None, None, None]

    process_input_fns: List[typings.ETL_fn] = [
        ignore_tensor(leave_first(
            ETL_Functions.load_npy_arr_from_dir,
            settings["data_dir"],
            settings["input_dir"],
            settings["img_size"],
            settings["tf_img_size"],
            0,
        )),
        ignore_key(leave_first(
            tf.math.divide,
            tf.cast(2 ** 16 - 1, dtype=tf.float32),
        )),
    ]

    process_target_fns = [
        ignore_tensor(leave_first(
            ETL_Functions.load_npy_arr_from_dir,
            settings["data_dir"],
            settings["label_dir"],
            settings["img_size"],
            settings["tf_img_size"],
            0,
        )),
        ignore_key(leave_first(
            tf.math.divide,
            tf.cast(65535, dtype=tf.float32)
        )),
    ]

    for idx, tup in enumerate([trn_idx, val_idx, test_idx]):
        start_idx, end_idx = tup
        criteria_arr = pic_num[start_idx:end_idx]

        etl = ETL_2D(
            dataset_json,
            norm_metadata,
            criteria_arr,
            settings["batch_size"],
            process_input_fns,
            process_target_fns,
        )
        datasets[idx] = etl.get_ml_dataset()

    trn_dataset = datasets[0]
    val_dataset = datasets[1]
    test_dataset = datasets[2]

    return (
        trn_dataset,
        val_dataset,
        test_dataset,
    )


if __name__ == "__main__":
    preprocess_ml_data()
