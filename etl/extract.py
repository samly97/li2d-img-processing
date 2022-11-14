from typing import Tuple, List, Dict

from utils import typings
from utils.io import save_numpy_arr
from utils.image import get_electrode_meshgrid
from utils.image import zoom_image
from utils.image import extract_input
from utils.metrics import measure_porosity

import os
from tqdm import tqdm

import numpy as np
from math import ceil


class SOLmap():

    def __init__(self, filepath: str):
        tup = self._get_exp_params(filepath)
        self.c_rate = tup[0]
        self.time = tup[1]
        self.solmap_arr = np.load(filepath)

    def __str__(self):
        return (
            "C-rate: %s\nTime (s): %s\n" % (self.c_rate, self.time)
        )

    def _get_exp_params(
        self,
        colmap_filepath: str
    ) -> Tuple[str, str]:
        r'''
        Returns: (c_rate, time)
        '''
        # Remove the full path, only the filename in the form of:
        # - c(c-rate)_t(time).png
        delimited = colmap_filepath.split('/')

        colmap_fname = delimited[-1]
        colmap_fname_split = colmap_fname.split('.npy')
        fname_scheme = colmap_fname_split[0]
        temp = fname_scheme.split('_t')

        # Get C-rate
        temp1 = temp[0]
        c_rate = temp1[1:]

        # Get timestep
        time = temp[1]

        return (c_rate, time)

##########################################################################
# BREAK ELECTRODE NUMPY ARRAYS INTO SUITABLE FORMAT FOR MACHINE LEARNING #
##########################################################################


class Microstructure_Breaker_Upper():

    r''' `Microstructure_Breaker_Upper` takes NumPy arrays which represents a
    voxelated representative of an electrode microstructure and a series of the
    voxelated SoLmaps genereated from `create_col_map.py` using the
    `Microstructure` class.

    Then, these data are processed to create input and output images amenable
    for training as well as "stand-alone" usage when using the trained Neural
    Network for predictions.

    Used in:
        - `create_ml_dataset.py` uses `ml_data_from_all_solmaps`
        - `postprocessing/ml.py` uses `extract_particles_from_microstructure`
    '''

    def __init__(
        self,
        micro_num: str,
        solmap_path: str,
        micro_arr: np.ndarray,
        L: int,
        h_cell: int,
        particles: List[typings.Circle_Info],
        sol_max: int,
        pore_encoding: int,
        padding_encoding: int,
        scale: int = 10,
    ):
        self.micro_num = micro_num
        self.solmap_path = solmap_path

        # Electrode properties
        self.L = L
        self.h_cell = h_cell
        self.particles = particles

        # User settings
        self.sol_max = sol_max
        self.pore_encoding = pore_encoding
        self.padding_encoding = padding_encoding
        self.scale = scale

        if self.particles is None or len(self.particles) == 0:
            raise ValueError(
                "Microstructure be a list of particles, but got %s",
                self.particles,
            )

        # Derived quantities
        self.micro_arr = micro_arr
        self.meshgrid = get_electrode_meshgrid(L, h_cell, scale)

        if self.solmap_path != "":
            self.sol_maps = self._get_solmaps()

    def _get_solmaps(self) -> List[SOLmap]:
        cm_dir = os.path.join(self.solmap_path, "col")
        # Assumes only colormap images in this directory
        files = os.listdir(cm_dir)
        files.sort()

        solmaps = [
            SOLmap(
                os.path.join(cm_dir, solmap)
            ) for solmap in files
        ]

        return solmaps

    def ml_data_from_all_solmaps(
        self,
        width_wrt_radius: int,
        output_img_size: int,
        input_dir: str,
        label_dir: str,
        pic_num: int,
    ) -> Tuple[int, Dict[str, typings.Metadata]]:
        r''' `ml_data_from_all_solmaps` extracts the input and output images as
        well as associated metadata, which is an intermediate step for creating
        the dataloader.

        This method is the format of data expected for training the Neural
        Network.
        '''

        ret_dict: Dict[str, typings.Metadata] = {}

        for solmap in tqdm(self.sol_maps):
            for particle in self.particles:

                (input_im,
                 label_im,
                 metadata
                 ) = _Extraction_Functionality.process_ml_data(
                    True,
                    self.micro_num, self.L,
                    particle, self.micro_arr, solmap,
                    width_wrt_radius,
                    output_img_size,
                    self.meshgrid,
                    self.pore_encoding, self.padding_encoding, self.sol_max,
                    self.scale,
                )

                # Insert metadata
                ret_dict[str(pic_num)] = metadata

                # Save input image
                input_fname = os.path.join(
                    input_dir,
                    str(pic_num) + ".npy",
                )
                save_numpy_arr(input_im, input_fname)

                # Save label image
                label_fname = os.path.join(
                    label_dir,
                    str(pic_num) + ".npy",
                )
                save_numpy_arr(label_im, label_fname)

                # Enumerate index
                pic_num += 1

        return pic_num, ret_dict

    def extract_particles_from_microstructure(
        self,
        width_wrt_radius: int,
        output_img_size: int,
    ) -> Tuple[np.ndarray, List[typings.Metadata]]:
        r''' `extract_particles_from_microstructure` extracts the input image
        and the metadata to create the dataloader. This is the expected format
        of data when using the trained Neural Network as a stand-alone
        evaluator.
        '''

        extracted_ims = np.zeros(
            (len(self.particles),
             output_img_size,
             output_img_size,
             1,
             ),
            dtype=np.uint16,
        )
        arr_meta = []

        for idx, particle in tqdm(enumerate(self.particles)):

            (circ_im, _, metadata) = _Extraction_Functionality.process_ml_data(
                False,
                "-1", self.L,
                particle, self.micro_arr, self.micro_arr,
                width_wrt_radius,
                output_img_size,
                self.meshgrid,
                self.pore_encoding, self.padding_encoding, self.sol_max,
                self.scale,
            )

            extracted_ims[idx] = circ_im
            arr_meta.append(metadata)

        return extracted_ims, arr_meta


class _Extraction_Functionality():

    r''' `_Extraction_Functionality` represents the key functionality in going
    from voxelized microstructure and State-of-Lithiation-map arrays into
    small, isolated images and metadata suitable for usage in a Machine
    Learning format.

    The steps are as follows:
    1.  `extract_input_and_cmap_im` is used to extract centered particles from
        the voxelized representations of the electrode and the SoLmap.
    2. `process_ml_data` is the main step in this extraction algorithm. The
        inputs to the Neural Network are created (input image and metadata) as
        well as the output image.

        This functionality is shared between training and stand-alone
        evaluation. In the latter, the output image is not needed, and setting
        `c_rate` and `time` in the metadata are handled separately in
        `postprocessing/ml.py`.
    '''

    @staticmethod
    def process_ml_data(
        for_training: bool,
        micro_num: str,
        L: int,
        particle: typings.Circle_Info,
        micro_arr: np.ndarray,
        solmap: SOLmap,
        width_wrt_radius: int,
        output_img_size: int,
        meshgrid: typings.meshgrid,
        pore_encoding: int,
        padding_encoding: int,
        sol_max: int,
        scale: int,
    ) -> Tuple[np.ndarray, np.ndarray, typings.Metadata]:

        R = particle["R"]
        # Size image according to radius factor
        pore_box_radius = ceil(float(R) * scale * width_wrt_radius)
        label_box_radius = ceil(float(R) * scale * width_wrt_radius)

        (input_im,
         label_im) = _Extraction_Functionality.extract_input_and_cmap_im(
            pore_box_radius,
            label_box_radius,
            micro_arr,
            solmap,
            particle,
            meshgrid,
            padding_encoding,
            scale,
            for_training=for_training,
        )

        if for_training:
            label_im = _Extraction_Helpers.scale_sol(label_im, sol_max)

        # Measure local porosity
        porosity = measure_porosity(
            input_im,
            np.array(pore_encoding),
            np.array(padding_encoding),
        )

        input_im, _ = zoom_image(input_im, output_img_size, order=0)
        label_im, zoom_factor = zoom_image(label_im, output_img_size, order=0)

        metadata: typings.Metadata = {
            "micro": micro_num,
            "x": float(particle["x"]) / L,
            "y": particle["y"],
            "R": particle["R"],
            "L": L,
            "zoom_factor": zoom_factor,
            "c_rate": "-1",
            "time": "-1",
            "dist_from_sep": float(particle["x"])/L,
            "porosity": porosity,
        }

        if for_training:
            metadata["c_rate"] = solmap.c_rate
            metadata["time"] = solmap.time

        return input_im, label_im, metadata

    @staticmethod
    def extract_input_and_cmap_im(
        pore_box_radius: int,
        label_box_radius: int,
        micro_arr: np.ndarray,
        solmap: SOLmap,
        particle: typings.Circle_Info,
        meshgrid: typings.meshgrid,
        padding_encoding: int,
        scale: int,
        for_training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r''' Gets both the input (blank) and labelled (with color) images to
        form machine learning data.

        Returns: (input_im, labelled_im)
        '''
        micro_im = np.copy(micro_arr)

        if for_training:
            sol_values = _Extraction_Helpers.get_sol_circle(
                solmap,
                particle,
                meshgrid,
            )

        input_im = extract_input(
            pore_box_radius,
            micro_im,
            (particle['x'], particle['y'], ""),
            scale,
            padding_encoding,
        )
        label_im = extract_input(
            label_box_radius,
            sol_values if for_training else micro_im,
            (particle['x'], particle['y'], ""),
            scale,
            padding=0,
        )

        return input_im, label_im


class _Extraction_Helpers():

    @staticmethod
    def scale_sol(
        label_im: np.ndarray,
        sol_max: int,
    ) -> np.ndarray:
        solid_phase = np.array(np.where(
            np.all(label_im != 0, axis=-1),
        ))
        ret_im = np.zeros_like(label_im, dtype=np.uint16)
        ret_im[solid_phase, :] = np.floor(
            label_im[solid_phase, :] * sol_max,
        )
        return ret_im

    @staticmethod
    def get_sol_circle(
        solmap: SOLmap,
        particle: typings.Circle_Info,
        meshgrid: typings.meshgrid,
    ) -> np.ndarray:
        r''' `get_sol_circle` retrieves a voxelated format of a SoLmap, where
        SoL values are superimposed on active particles. This returns SoL
        values on the same SoLmap, but with all other particles zeroed out.'''

        ret_im = np.copy(solmap.solmap_arr)
        xx, yy = meshgrid

        x = float(particle["x"])
        y = float(particle["y"])
        r = float(particle["R"])

        # Boolean array where "True" is a pixel inside the circle of analysis
        in_circ = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) <= r

        # "Turn off" all pixels outside of current circle
        # Note: can be problematic if we start with 0 for concentration...
        # May be better to use the pore-phase encoding value.
        ret_im[~in_circ] = 0
        return ret_im
