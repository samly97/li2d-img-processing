
import os
from typing import List, Dict, Tuple, Callable, Union

import tensorflow as tf
import numpy as np

from scipy.ndimage import zoom
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import porespy as ps

from utils import typings


class ETL_Functions():

    @staticmethod
    def load_npy_arr_from_dir(
        pic_num: int,
        data_dir: str,
        img_dir: str,
        img_size: int,
        tf_img_size: int,
        order: int = 0,
    ) -> tf.types.experimental.TensorLike:
        path = tf.strings.join(
            [data_dir, "/", img_dir]
        )
        fname = tf.strings.join(
            [path, "/", tf.strings.as_string(pic_num), ".npy"])

        img = tf.numpy_function(
            np.load, [fname], tf.uint16,
        )
        img = tf.cast(img, dtype=tf.float32)

        zoom_factor = tf_img_size / img_size
        img = tf.py_function(
            lambda a: zoom(a, (zoom_factor, zoom_factor, 1), order=order),
            [img],
            tf.float32,
        )
        return img

    @ staticmethod
    def gather_img_from_np_arr(
        idx_num: int,
        arr_imgs: np.ndarray,
        img_size: int,
        tf_img_size: int,
        order: int = 0,
    ) -> tf.types.experimental.TensorLike:
        img = tf.gather(arr_imgs, idx_num)
        img = tf.convert_to_tensor(img)

        zoom_factor = tf_img_size / img_size
        img = tf.py_function(
            lambda a: zoom(a, (zoom_factor, zoom_factor, 1), order=order),
            [img],
            tf.float32,
        )
        return img

    @ staticmethod
    def get_mask(
        idx_num: int,
        inp_img: tf.types.experimental.TensorLike,
        tf_img_size: int,
        encoding: float = 1.0,
    ):
        center_loc = tf_img_size // 2

        # Threshold input image to boolean where `True` is the solid phase. The
        # solid phase is taken to be less than the pore encoding as greater
        # than the padding encoding of `0.0`.
        cond1 = tf.math.less(inp_img, encoding)
        cond2 = tf.math.greater(inp_img, 0.0)

        thres = tf.math.logical_and(cond1, cond2)
        thres = tf.cast(thres, tf.bool)

        # Use the EDT on the solid phase with `True` values
        distance = tf.py_function(
            ndi.distance_transform_edt, [thres], tf.float32)

        # Return array where `True` entries represent the peaks
        peaks = tf.numpy_function(
            ps.filters.find_peaks, [distance], tf.bool,
        )

        # Label the `peaks` in an array with unique integer numbers for the
        # peaks
        markers, _ = tf.numpy_function(
            ndi.label, [peaks], [tf.int32, tf.int64],
        )

        # Take the negative of distance since watershed fills "basins"
        distance = tf.math.scalar_mul(-1., distance)
        # Watershed segmentation
        labels = tf.numpy_function(
            watershed, [distance, markers], tf.int32,
        )

        # Find label of the center particle and then do a boolean
        # for the center image
        center_lab = labels[center_loc, center_loc]

        # Get indices in the array which correspond to labels for the center
        # particle
        label_indices = tf.where(
            tf.math.equal(labels, center_lab)
        )

        # Update a boolean image where `True` correspond to the watershed
        # region for the particle of interest
        shape = tf.shape(label_indices)
        update = tf.ones(shape[0], tf.bool)

        mask_im = tf.zeros_like(thres, tf.bool)
        mask_im = tf.tensor_scatter_nd_update(
            mask_im, label_indices, update,
        )

        return tf.math.logical_and(mask_im, thres)

    @ staticmethod
    def format_metadata(
        idx_num: int,
        metadata_lookup_table: tf.lookup.StaticHashTable,
    ) -> tf.types.experimental.TensorLike:
        key = tf.strings.as_string(idx_num)
        as_str = metadata_lookup_table[key]
        str_nums = tf.strings.split(as_str, sep="-")
        ret = tf.strings.to_number(str_nums, out_type=tf.dtypes.float32)
        return ret

    @ staticmethod
    def fake_output(
        idx_num: int,
        tf_img_size: int,
    ) -> tf.types.experimental.TensorLike:
        "Note: may not work... may need to wrap it in tf.numpy_fn"
        _ = idx_num
        blank_im = np.zeros(
            (tf_img_size, tf_img_size, 1)
        )
        blank_im = tf.convert_to_tensor(blank_im)
        return blank_im


class ETL_2D():

    def __init__(
        self,
        metadata: Dict[str, typings.Metadata],
        metadata_norm: typings.Metadata_Normalizations,
        criteria_arr: List[int],
        batch_size: int,
        tf_img_size: int,
        process_input_fns: List[typings.ETL_fn],
        process_target_fns: List[typings.ETL_fn],
    ):
        self.AUTOTUNE = tf.data.AUTOTUNE
        self.metadata_norm = metadata_norm
        self.batch_size = batch_size
        self.tf_img_size = tf_img_size

        self.metadata_lookup = self.get_static_hash_table(
            metadata,
            process_value_fn=lambda num: self._format_metadata(
                num,
                metadata,
            ),
            default_value="",
        )

        self.starting_ds = tf.data.Dataset.from_tensor_slices(criteria_arr)

        self.input_fns = process_input_fns
        self.output_fns = process_target_fns

    def get_ml_dataset(self) -> tf.data.Dataset:

        def configure_for_performance(ds):
            # https://www.tensorflow.org/tutorials/load_data/images#using_tfdata_for_finer_control
            ds = ds.cache()
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(buffer_size=self.AUTOTUNE)
            return ds

        inp_ds = self.starting_ds.map(
            self._process_input_path, num_parallel_calls=self.AUTOTUNE)
        out_ds = self.starting_ds.map(
            self._process_output_path, num_parallel_calls=self.AUTOTUNE)

        inp_ds = configure_for_performance(inp_ds)
        out_ds = configure_for_performance(out_ds)

        ret_ds = tf.data.Dataset.zip((inp_ds, out_ds))
        return ret_ds

    def _process_input_path(
        self,
        arr_idx: int,
        input_im: tf.types.experimental.TensorLike = None,
    ) -> Tuple[
            tf.types.experimental.TensorLike,
            tf.types.experimental.TensorLike,
            tf.types.experimental.TensorLike,
    ]:
        for fn in self.input_fns:
            input_im = fn(arr_idx, input_im)

        mask_im = ETL_Functions.get_mask(
            arr_idx, input_im, self.tf_img_size,
        )
        metadata = ETL_Functions.format_metadata(
            arr_idx,
            self.metadata_lookup,
        )

        return input_im, mask_im, metadata

    def _process_output_path(
        self,
        arr_idx: int,
        target_im: tf.types.experimental.TensorLike = None,
    ) -> tf.types.experimental.TensorLike:
        for fn in self.output_fns:
            target_im = fn(arr_idx, target_im)

        return target_im

    def get_static_hash_table(
        self,
        hash,
        process_value_fn: Callable[[int], str],
        default_value: str = "",
    ) -> tf.lookup.StaticHashTable:
        r''' The default value depends on what the data type of the key is.
        '''
        keys: List[Union[str, None]] = [None for _ in range(0, len(hash))]
        vals: List[Union[str, None]] = [None for _ in range(0, len(hash))]

        for i, key in enumerate(hash.keys()):
            val = process_value_fn(key)

            keys[i] = str(key)
            vals[i] = val

        keys = tf.constant(keys)
        vals = tf.constant(vals)

        ret_table = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(keys, vals),
            default_value=default_value,
        )

        return ret_table

    def _format_metadata(
        self,
        pic_num: int,
        metadata: Dict[str, typings.Metadata],
    ) -> str:

        hash = metadata[str(pic_num)]

        x = float(hash["x"])
        y = float(hash["y"]) / self.metadata_norm["h_cell"]
        R = float(hash["R"]) / self.metadata_norm["R_max"]
        L = float(hash["L"]) / self.metadata_norm["L"]
        zoom = float(hash["zoom_factor"]) / self.metadata_norm["zoom_norm"]
        c_rate = float(hash["c_rate"]) / self.metadata_norm["c_rate_norm"]
        time = float(hash["time"]) / self.metadata_norm["time_norm"]
        porosity = float(hash["porosity"])
        dist_from_sep = float(hash["dist_from_sep"])

        as_float = [x, y, R, L, zoom, c_rate, time, porosity, dist_from_sep]

        s = "-"
        ret = s.join(str(num) for num in as_float)
        return ret

###############################################################################
# DECORATORS ##################################################################
###############################################################################


def leave_first(fn, *args):
    def wrapped(
        first_arg: int,
    ) -> typings.ETL_key_fn:
        return fn(first_arg, *args)
    return wrapped


def ignore_key(
    fn: typings.ETL_key_fn,
) -> typings.ETL_key_and_tensor_fn:
    def wrapped(arr_idx, tensor):
        _ = arr_idx
        return fn(tensor)
    return wrapped


def ignore_tensor(
    fn: typings.ETL_key_fn
) -> typings.ETL_key_and_tensor_fn:
    def wrapped(arr_idx, tensor):
        _ = tensor
        return fn(arr_idx)
    return wrapped
