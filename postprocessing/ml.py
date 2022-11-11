from typing import Tuple, List, Dict

import numpy as np
import tensorflow as tf

from utils.numerics import get_coords_in_circle
from utils.numerics import get_inscribing_meshgrid
from utils.image import zoom_image

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
        user_params: Tuple[int, int, int, int],
    ):
        micro_fname, cell_length, cell_height = micro_data
        width_wrt_radius, scale, output_img_size, order = user_params

        # Electrode settings/parameters
        self.micro_fname = micro_fname
        self.cell_length = cell_length
        self.cell_height = cell_height
        self.circle_data = circle_data

        # Image settings
        self.width_wrt_radius = width_wrt_radius
        self.scale = scale
        self.output_img_size = output_img_size
        self.order = order

        # Dataset settings
        self.tf_img_size = tf_img_size
        self.batch_size = batch_size
        self.norm_metadata = norm_metadata

        # This only runs during instantiation. Extracting images is
        # computationally expensive.
        arr_imgs, arr_meta = self._prep_micro_data(
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
    ) -> tf.data.Dataset:
        arr_meta = self._set_rate_and_time(c_rate, time, self.arr_meta)
        hash_meta = self._meta_arr_to_hash(arr_meta)

        process_input_fns = [
            ignore_tensor(leave_first(
                ETL_Functions.gather_img_from_np_arr,
                self.arr_imgs,
                self.output_img_size,
                self.tf_img_size,
                self.order,
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
            process_input_fns,
            process_target_fns,
        )
        dataset_loader = etl.get_ml_dataset()

        return dataset_loader

    def _prep_micro_data(
        self,
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
            scale,
        )

        arr_imgs, arr_meta = mbu.extract_particles_from_microstructure(
            width_wrt_radius,
            output_img_size,
        )

        return arr_imgs, arr_meta

    def _set_rate_and_time(
        self,
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

    def _meta_arr_to_hash(
        self,
        arr_meta: List[typings.Metadata],
    ) -> Dict[str, typings.Metadata]:
        ret = {}
        for idx, hash in enumerate(arr_meta):
            ret[str(idx)] = hash
        return ret


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
