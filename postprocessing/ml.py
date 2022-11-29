from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf

import scipy.ndimage as ndi

from etl.etl import ETL_2D
from etl.etl import ETL_Functions
from etl.etl import leave_first, ignore_key, ignore_tensor
from etl.extract import Microstructure_Breaker_Upper

from utils import typings

######################################################
# "BREAK" ELECTRODE IN ML-"LEARNABLE" IMAGES/DATASET #
######################################################


class Microstructure_to_ETL:
    r''' Microstructure_to_ETL takes a 2D electrode microstructure with
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

    def __init__(
        self,
        tf_img_size: int,
        batch_size: int,
        norm_metadata: typings.Metadata_Normalizations,
        circle_data: List[typings.Circle_Info],
        micro_data: Tuple[str, int, int],
        pore_encoding: int,
        padding_encoding: int,
        user_params: Tuple[int, int, int],
    ):
        micro_fname, cell_length, cell_height = micro_data
        width_wrt_radius, scale, output_img_size = user_params

        # Electrode settings/parameters
        self.micro_fname = micro_fname
        self.cell_length = cell_length
        self.cell_height = cell_height
        self.circle_data = circle_data

        # Image settings
        self.width_wrt_radius = width_wrt_radius
        self.scale = scale
        self.output_img_size = output_img_size

        # Dataset settings
        self.tf_img_size = tf_img_size
        self.batch_size = batch_size
        self.norm_metadata = norm_metadata

        # This only runs during instantiation. Extracting images is
        # computationally expensive.
        arr_imgs, arr_meta = _Create_ETL_Helpers.prep_micro_data(
            self.circle_data,
            micro_data,
            user_params,
            pore_encoding,
            padding_encoding
        )

        self.arr_imgs = arr_imgs
        self.arr_meta = arr_meta

    def get_loader(
        self,
        c_rate: float,
        time: int,
        cache=None,
    ) -> tf.data.Dataset:
        arr_meta = _Create_ETL_Helpers.set_rate_and_time(
            c_rate,
            time,
            self.arr_meta
        )
        hash_meta = _Create_ETL_Helpers.meta_arr_to_hash(arr_meta)

        process_input_fns = [
            ignore_tensor(leave_first(
                ETL_Functions.gather_img_from_np_arr,
                self.arr_imgs,
                self.output_img_size,
                self.tf_img_size,
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
                self.tf_img_size,
            ))
        ]

        etl = ETL_2D(
            hash_meta,
            self.norm_metadata,
            [i for i in range(len(self.circle_data))],
            self.batch_size,
            self.tf_img_size,
            process_input_fns,
            process_target_fns,
        )

        if cache is None:
            dataset_loader = etl.get_ml_dataset()

            inputs = dataset_loader.map(lambda ins, _: ins)
            targets = dataset_loader.map(lambda _, targets: targets)

            edt_ims = inputs.map(lambda edt, mask, _: edt)
            mask_ims = inputs.map(lambda edt, mask, _: mask)

            cache = {"edt": edt_ims, "mask": mask_ims, "target": targets}

        else:
            dataset_loader = etl.amend_metadata_to_loader(
                cache["edt"], cache["mask"], cache["target"],
            )

        return dataset_loader, cache


class _Create_ETL_Helpers():

    @staticmethod
    def prep_micro_data(
        circle_data: List[typings.Circle_Info],
        micro_data: Tuple[str, int, int],
        user_params: Tuple[int, int, int],
        pore_encoding: int,
        padding_encoding: int,
    ) -> Tuple[np.ndarray, List[typings.Metadata]]:
        r''' Extracts each from an electrode microstructure, centering it with a
        constant "box radius" as well as preparing the metadata.
        '''
        micro_fname, cell_length, cell_height = micro_data
        width_wrt_radius, scale, output_img_size = user_params

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
            scale,
        )

        arr_imgs, arr_meta = mbu.extract_particles_from_microstructure(
            width_wrt_radius,
            output_img_size,
        )

        return arr_imgs, arr_meta

    @staticmethod
    def set_rate_and_time(
        c_rate: float,
        time: float,
        arr_meta: List[typings.Metadata],
    ) -> List[typings.Metadata]:
        ret: List[typings.Metadata] = [{
            "micro": "",
            "x": -1,
            "y": "",
            "R": "",
            "L": -1,
            "zoom_factor": -1,
            "c_rate": "",
            "time": "",
            "dist_from_sep": -1,
            "porosity": -1,
        } for _ in range(len(arr_meta))]

        for idx, hash in enumerate(arr_meta):
            hash["c_rate"] = str(c_rate)
            hash["time"] = str(time)

            ret[idx] = hash

        return ret

    @staticmethod
    def meta_arr_to_hash(
        arr_meta: List[typings.Metadata],
    ) -> Dict[str, typings.Metadata]:
        ret = {}
        for idx, hash in enumerate(arr_meta):
            ret[str(idx)] = hash
        return ret

######################################################
# PREDICT "SOLMAP" FROM MACHINE LEARNING PREDICTIONS #
######################################################


def electrode_sol_map_from_predictions(
    input_dataset: tf.data.Dataset,
    predicted_imgs: np.ndarray,
    L_electrode: int,
    norm_metadata: Tuple[int, int, int, float, int, int],
    batch_size: int,
    scale: int = 10,
) -> np.ndarray:
    r'''`electrode_sol_map_from_predictions` takes an electrode (in the form of
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

    tf.experimental.numpy.experimental_enable_numpy_behavior()

    _, h_cell, _, zoom_norm, _, _ = norm_metadata

    solmap = np.zeros(
        (h_cell * scale, L_electrode * scale, 1),
        dtype=np.float32,
    )

    predicted_imgs = tf.data.Dataset.from_tensor_slices(predicted_imgs)
    predicted_imgs = predicted_imgs.batch(batch_size)

    # For each input (image, mask, metadata) in the dataset, extract the color
    # from the Machine Learning output and place it in the coordinate in the
    # electrode.
    for data_batch, pred_batch in zip(input_dataset, predicted_imgs):

        in_batch, _ = data_batch
        _, _, meta_batch = in_batch

        x_centers = _ML_Pred_to_Solmap.get_meta_elem("x", meta_batch) * \
            tf.cast(L_electrode, tf.float32)
        y_centers = _ML_Pred_to_Solmap.get_meta_elem("y", meta_batch) * \
            tf.cast(h_cell, tf.float32)
        zoom_factors = _ML_Pred_to_Solmap.get_meta_elem(
            "zoom_factor", meta_batch) * tf.cast(zoom_norm, tf.float32)

        # Zoom images back to their original sizes
        unzoomed_imgs = tf.map_fn(
            _ML_Pred_to_Solmap.zoom_tensor_ret_img,
            (pred_batch, zoom_factors),
            fn_output_signature=tf.RaggedTensorSpec(
                shape=None,
                ragged_rank=1,
                dtype=tf.float32,
            )
        )

        # Get the center pixel of a prediction image
        prediction_centers = tf.map_fn(
            lambda im: tf.math.scalar_mul(
                tf.cast(1/2, tf.float32),
                tf.cast(tf.shape(im.to_tensor())[0], tf.float32),
            ),
            unzoomed_imgs,
            fn_output_signature=tf.TensorSpec(
                shape=(),
                dtype=tf.float32,
            )
        )

        # Get a tensor of masks for the SoL values (RaggedTensor)
        sol_value_mask = tf.math.greater(unzoomed_imgs, 0.0)

        for batch_idx in range(pred_batch.shape[0]):
            x = x_centers[batch_idx]
            y = y_centers[batch_idx]
            center = prediction_centers[batch_idx]

            zoomed_pred_img = unzoomed_imgs[batch_idx].to_tensor()

            # Meshgrids where a non-zero pixel is the x or y coordinate of a
            # pixel which has a predicted SoL value.
            sol_X, sol_Y = _ML_Pred_to_Solmap.sol_pixel_meshgrid(
                sol_value_mask[batch_idx])

            # Translate the meshgrids so we are:
            #   - subtract predicted image center (circle centered at (0, 0))
            #   - add the scaled (x, y) center particle coordinates
            elec_X, elec_Y = _ML_Pred_to_Solmap.sol_to_electrode_meshgrid(
                (sol_X, sol_Y, sol_value_mask[batch_idx], x, y, center),
                scale,
            )

            # Take SoL values from the "unzoomed" prediction images and place
            # it in the corresponding meshgrid location in the SoLmap
            solmap[
                elec_Y,
                elec_X,
                :,
            ] = zoomed_pred_img.numpy()[sol_Y, sol_X, :]

    return solmap


class _ML_Pred_to_Solmap():

    @ staticmethod
    def get_meta_elem(
        elem: str,
        meta_tensor: tf.types.experimental.TensorLike
    ):
        return tf.gather(
            meta_tensor,
            typings.META_INDICES[elem],
            axis=1,
        )

    @ staticmethod
    def zoom_tensor_ret_img(inputs):
        img, zoom_factor = inputs

        out_img_size = 1 / zoom_factor

        unzoomed_img = tf.py_function(
            lambda arr: ndi.zoom(
                arr, (out_img_size, out_img_size, 1), order=0
            ),
            [img], tf.float32,
        )

        unzoomed_img = tf.RaggedTensor.from_tensor(unzoomed_img)
        return unzoomed_img

    @ staticmethod
    def sol_pixel_meshgrid(sol_value_mask):
        # Get the pixel location of where SoL values are in the direct Machine
        # Learning predictions

        zoomed_im = sol_value_mask.to_tensor()
        zoomed_img_size = tf.shape(zoomed_im)[0]

        # Linearly spaced array
        img_linspace = tf.range(0, zoomed_img_size)

        # Meshgrid with coordinates corresponding to `zoomed_im` size
        X, Y = tf.meshgrid(img_linspace, img_linspace)

        def fn(T): return tf.math.multiply(
            T, tf.cast(zoomed_im[:, :, 0], tf.int32))

        X = fn(X)
        Y = fn(Y)

        return X, Y

    @ staticmethod
    def sol_to_electrode_meshgrid(tup, scale):
        # There may be some controversy in choosing which pixel would be the
        # center either using `ceil`, `floor`, or `round` approach.
        #
        # For odd-sized images this is easy, it's just 1/2 the image rounded
        # down, but for evenly-sized images this might be controversial:
        #   5-pixels:
        #       5 / 2 = 2.5 ==> 2 ([] [] [x] [] [])
        #   4-pixels:
        #       4 / 2 = 2   ==> 2 ([] [x] [] [])

        X, Y, sol_value_mask, x, y, center = tup

        sol_value_mask = sol_value_mask.to_tensor()

        def fn(T, coord):
            return tf.cast(sol_value_mask, tf.float32) * scale * coord + \
                (tf.cast(tf.expand_dims(T, axis=-1), tf.float32) - center)

        # Translate SoL meshgrid to its location in the electrode
        elec_X = tf.cast(tf.math.round(fn(X, x)), tf.int64)
        elec_Y = tf.cast(tf.math.round(fn(Y, y)), tf.int64)

        # Reshape since meshgrid should be (dim, dim) and not (dim, dim, 1)
        elec_X = elec_X[:, :, 0]
        elec_Y = elec_Y[:, :, 0]

        return elec_X, elec_Y
