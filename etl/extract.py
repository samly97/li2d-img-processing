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


class Microstructure_Breaker_Upper():
    # Seems like we could extend the `Microstructure` class from the
    # `create_col_map.py` file... I could see how there's similarities... food
    # for thought.
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
        ret_dict: Dict[str, typings.Metadata] = {}

        for solmap in tqdm(self.sol_maps):
            for particle in self.particles:
                input_im, label_im, metadata = self.ml_data_from_particles(
                    particle,
                    solmap,
                    width_wrt_radius,
                    output_img_size,
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

    def ml_data_from_particles(
        self,
        particle: typings.Circle_Info,
        solmap: SOLmap,
        width_wrt_radius: int,
        output_img_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, typings.Metadata]:
        R = particle["R"]
        # Size image according to radius factor
        pore_box_radius = ceil(float(R) * self.scale * width_wrt_radius)
        label_box_radius = ceil(float(R) * self.scale)

        input_im, label_im = self.extract_input_and_cmap_im(
            pore_box_radius,
            label_box_radius,
            solmap,
            particle,
        )
        label_im = self.scale_sol(label_im)

        # Measure local porosity
        porosity = measure_porosity(
            input_im,
            np.array(self.pore_encoding),
            np.array(self.padding_encoding),
        )

        input_im, _ = zoom_image(input_im, output_img_size, order=0)
        label_im, zoom_factor = zoom_image(label_im, output_img_size, order=0)

        metadata: typings.Metadata = {
            "micro": self.micro_num,
            "x": float(particle["x"]) / self.L,
            "y": particle["y"],
            "R": particle["R"],
            "L": self.L,
            "zoom_factor": zoom_factor,
            "c_rate": solmap.c_rate,
            "time": solmap.time,
            "dist_from_sep": float(particle["x"])/self.L,
            "porosity": porosity,
        }

        return input_im, label_im, metadata

    def extract_particles_from_microstructure(
        self,
        width_wrt_radius: int,
        output_img_size: int,
        order: int,
    ) -> Tuple[np.ndarray, List[typings.Metadata]]:
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
            circ_im, metadata = self._extract_particle_from_microstructures(
                particle,
                width_wrt_radius,
                output_img_size,
                order,
            )

            extracted_ims[idx] = circ_im
            arr_meta.append(metadata)

        return extracted_ims, arr_meta

    def _extract_particle_from_microstructures(
        self,
        particle: typings.Circle_Info,
        width_wrt_radius: int,
        output_img_size: int,
        order: int,
    ) -> Tuple[np.ndarray, typings.Metadata]:
        R = particle["R"]
        # Size image according to radius factor
        pore_box_radius = ceil(float(R) * self.scale * width_wrt_radius)
        label_box_radius = ceil(float(R) * self.scale)

        micro_arr = np.copy(self.micro_arr)
        dummy = np.copy(micro_arr)

        circ_im = extract_input(
            pore_box_radius,
            micro_arr,
            (particle['x'], particle['y'], ""),
            self.scale,
            self.padding_encoding,
        )
        dummy_im = extract_input(
            label_box_radius,
            dummy,
            (particle['x'], particle['y'], ""),
            self.scale,
            self.padding_encoding,
        )

        porosity = measure_porosity(
            circ_im,
            np.array(self.pore_encoding),
            np.array(self.padding_encoding),
        )

        zoomed_circ_im, _ = zoom_image(
            circ_im,
            output_img_size,
            order=order,
        )
        _, zoom_factor = zoom_image(
            dummy_im,
            output_img_size,
            order=order,
        )

        circ_meta: typings.Metadata = {
            'micro': '-1',
            'x': float(particle["x"]) / self.L,
            'y': particle["y"],
            'R': particle["R"],
            'L': self.L,
            'zoom_factor': zoom_factor,
            'c_rate': '-1',
            'time': '-1',
            'porosity': porosity,
            'dist_from_sep': float(particle["x"]) / self.L,
        }

        return zoomed_circ_im, circ_meta

    def extract_input_and_cmap_im(
        self,
        pore_box_radius: int,
        label_box_radius: int,
        solmap: SOLmap,
        particle: typings.Circle_Info,
    ) -> Tuple[np.ndarray, np.ndarray]:
        r''' Gets both the input (blank) and labelled (with color) images to
        form machine learning data.

        Returns: (input_im, labelled_im)
        '''
        micro_im = np.copy(self.micro_arr)
        sol_values = self._get_sol_circle(
            solmap,
            particle,
        )

        input_im = extract_input(
            pore_box_radius,
            micro_im,
            (particle['x'], particle['y'], ""),
            self.scale,
            self.padding_encoding,
        )
        label_im = extract_input(
            label_box_radius,
            sol_values,
            (particle['x'], particle['y'], ""),
            self.scale,
            padding=0,
        )

        return input_im, label_im

    def scale_sol(
        self,
        label_im: np.ndarray
    ) -> np.ndarray:
        solid_phase = np.array(np.where(
            np.all(label_im != 0, axis=-1),
        ))
        ret_im = np.zeros_like(label_im, dtype=np.uint16)
        ret_im[solid_phase, :] = np.floor(
            label_im[solid_phase, :] * self.sol_max,
        )
        return ret_im

    def _get_sol_circle(
        self,
        solmap: SOLmap,
        particle: typings.Circle_Info,
    ) -> np.ndarray:
        ret_im = np.copy(solmap.solmap_arr)
        xx, yy = self.meshgrid

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