import os
from typing import List, Tuple
from math import ceil
import numpy as np
from random import shuffle
import tensorflow as tf

from utils.io import load_json


def parse_raw_data(
    data_dir: str,
    input_dir: str,
) -> List[str]:

    data_dir = os.path.join(os.getcwd(), data_dir)
    input_im_dir = os.path.join(data_dir, input_dir)
    input_im_fname = np.array([fname for fname in os.listdir(input_im_dir)])

    pic_num = [fname.split(".png")[0] for fname in input_im_fname]

    return pic_num


def shuffle_dataset(
    pic_num: List[str],
) -> np.array:

    # Shuffle list
    list_idx = list(range(len(pic_num)))
    shuffle(list_idx)

    # Shuffle particle number list
    pic_num = np.array(pic_num)
    pic_num = pic_num[list_idx]

    return pic_num


def get_split_indices(
    trn_split: int,
    val_split: int,
    pic_num: np.array,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:

    trn_idx = ceil(len(pic_num) * trn_split)
    val_idx = ceil(len(pic_num) * val_split) + trn_idx
    test_idx = len(pic_num)

    return ((0, trn_idx), (trn_idx, val_idx), (val_idx, test_idx))


def get_ml_dataset(
    pic_num: np.array,
    dataset_json,
    im_size: int,
    batch_size: int,
    norm_metadata: Tuple[int, int, int, int, int],
    dirs: Tuple[str, str, str],
    start_idx: int,
    end_idx: int,
):
    AUTOTUNE = tf.data.AUTOTUNE

    data_dir, input_dir, label_dir = dirs

    new_data = _get_static_hash_table(
        dataset_json,
        val_fn=lambda num: format_metadata(
            num,
            dataset_json,
            norm_metadata,
        ),
        default_value="",
    )

    fname_ds = tf.data.Dataset.from_tensor_slices(pic_num[start_idx: end_idx])

    def process_path_inputs(pic_num):
        input_im = _load_image(
            pic_num,
            data_dir,
            input_dir,
            im_size,
        )
        metadata = _format_metadata(
            pic_num,
            new_data,
        )

        return input_im, metadata

    def process_path_targets(pic_num):
        label_im = _load_image(
            pic_num,
            data_dir,
            label_dir,
            im_size,
        )

        return label_im

    def configure_for_performance(ds):
        # https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control
        ds = ds.cache()
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    inp_ds = fname_ds.map(process_path_inputs, num_parallel_calls=AUTOTUNE)
    out_ds = fname_ds.map(process_path_targets, num_parallel_calls=AUTOTUNE)

    inp_ds = configure_for_performance(inp_ds)
    out_ds = configure_for_performance(out_ds)

    ret_ds = tf.data.Dataset.zip((inp_ds, out_ds))
    return ret_ds


def _get_static_hash_table(
        hash: dict,
        val_fn=lambda v: v,
        default_value="") -> tf.lookup.StaticHashTable:
    r''' The default value depends on what the data type of the key is.
    '''

    keys = [None for _ in range(0, len(hash))]
    vals = [None for _ in range(0, len(hash))]

    for i, key in enumerate(hash.keys()):
        val = val_fn(key)

        keys[i] = str(key)
        vals[i] = val

    keys = tf.constant(keys)
    vals = tf.constant(vals)

    ret_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, vals),
        default_value=default_value)

    return ret_table


def format_metadata(
        pic_num: str,
        dataset_json,
        norm_metadata: Tuple[int, int, int, int, int]) -> List[str]:
    r''' format_metadata is the step prior to processing data into a
    tf.data.Dataset used for the Machine Learning pipeline. Due to TensorFlow
    not behaving well with Dictionaries or Lists in their built-in lookup
    tables, this step returns the metadata as strings. In the pipeline step,
    the strings are split and parsed into floats.
    '''

    L, h_cell, R_max, zoom_norm, c_rate_norm, time_norm = norm_metadata

    hash = dataset_json[pic_num]

    x = float(hash["x"]) / L
    y = float(hash["y"]) / h_cell
    R = float(hash["R"]) / R_max
    zoom = float(hash["zoom_factor"]) / zoom_norm
    c_rate = float(hash["c-rate"]) / c_rate_norm
    time = float(hash["time"]) / time_norm
    porosity = float(hash["porosity"])
    dist_from_sep = float(hash["dist_from_sep"])

    as_float = [x, y, R, zoom, c_rate, time, porosity, dist_from_sep]

    s = "-"
    ret = s.join([str(num) for num in as_float])
    return ret


def _load_image(
    file_num,
    data_dir: str,
    img_dir: str,
    im_size: int,
):
    path = tf.strings.join([os.getcwd(), "/", data_dir, "/", img_dir])

    img = tf.io.read_file(
        tf.strings.join([path, "/", file_num, ".png"])
    )
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, (im_size, im_size))
    return img


def _format_metadata(
    pic_num: tf.Tensor,
    metadata_tensor: tf.lookup.StaticHashTable,
):
    as_str = metadata_tensor[pic_num]
    str_nums = tf.strings.split(as_str, sep="-")
    ret = tf.strings.to_number(str_nums, out_type=tf.dtypes.float32)
    return ret


def preprocess_ml_data() -> Tuple[
        tf.data.Dataset,
        tf.data.Dataset,
        tf.data.Dataset]:
    r''' Define alternate entry point to when the script is not being called
    explicitly, i.e., cases other than "__name__ == __main__".
    '''
    settings = load_json("pipeline.json")

    norm_metadata = (
        settings["L"],
        settings["h_cell"],
        settings["R_max"],
        settings["zoom_max"],
        settings["c-rate"],
        settings["time"],
    )

    dataset_json = load_json(
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

    trn_dataset = get_ml_dataset(
        pic_num,
        dataset_json,
        settings["im_size"],
        settings["batch_size"],
        norm_metadata,
        (settings["data_dir"],
         settings["input_dir"],
         settings["label_dir"]),
        start_idx=trn_idx[0],
        end_idx=trn_idx[1],
    )

    val_dataset = get_ml_dataset(
        pic_num,
        dataset_json,
        settings["im_size"],
        settings["batch_size"],
        norm_metadata,
        (settings["data_dir"],
         settings["input_dir"],
         settings["label_dir"]),
        start_idx=val_idx[0],
        end_idx=val_idx[1],
    )

    test_dataset = get_ml_dataset(
        pic_num,
        dataset_json,
        settings["im_size"],
        settings["batch_size"],
        norm_metadata,
        (settings["data_dir"],
         settings["input_dir"],
         settings["label_dir"]),
        start_idx=test_idx[0],
        end_idx=test_idx[1],
    )

    return (
        trn_dataset,
        val_dataset,
        test_dataset,
    )


if __name__ == "__main__":
    """
    Copy-paste form preprocess_ml_data if desired to test this script:

    Run in terminal and diagnose:
    - python preprocess_ml_data.py
    """
    pass
