from typing import List, Dict, Callable, Tuple

from utils import typings

from utils.io import save_numpy_arr

from utils.numerics import get_inscribing_meshgrid
from utils.numerics import get_inscribing_coords
from utils.numerics import get_coords_in_circle

from utils.image import electrode_mask_2D

from preprocessing.data import Experiment
from preprocessing.data import COMSOL_Electrochem_CSV

import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


class Microstructure():

    m_to_um: float = 1e6

    def __init__(
        self,
        csv_formatter: COMSOL_Electrochem_CSV,
        micro_path: str,
        L: int,
        h_cell: int,
        c_rates: List[str],
        particles: List[typings.Circle_Info],
        grid_size: int = 1000,
        scale: int = 10,
    ):
        self.csv_formatter = csv_formatter

        self.micro_path = micro_path

        self.L = L
        self.h_cell = h_cell
        self.c_rates = c_rates

        self.particles = particles

        self.grid_size = grid_size
        self.scale = scale

        if self.c_rates is None or len(self.c_rates) == 0:
            raise ValueError(
                "c-rates should be a list of discharges, but got %s",
                self.c_rates,
            )

        if self.particles is None or len(self.particles) == 0:
            raise ValueError(
                "Microstructure be a list of particles, but got %s",
                self.particles,
            )

        self.experiments: Dict[str, Experiment] = self._get_experiments()
        self.electrode_mask = electrode_mask_2D(
            self.particles,
            self.L,
            self.h_cell,
            self.scale,
        )

    def __str__(self):
        return (
            'Electrode Length (um): %d\n' +
            'Electrode Width  (um): %d\n' +
            'C-rates: %s\n' +
            'Number of particles: %d\n'
        ) % (self.L, self.h_cell, self.c_rates, len(self.particles))

    def _get_experiments(self) -> Dict[str, Experiment]:
        experiments = {}

        for c_rate in self.c_rates:
            dataframe = self.csv_formatter.read_csv_into_pandas(
                self.micro_path,
                c_rate
            )
            exp = Experiment(
                c_rate,
                dataframe,
                scale=self.scale,
            )
            experiments[c_rate] = exp

        return experiments

    def create_and_save_all_colormaps_from_experiments(
        self,
        fname_fn: Callable[[str, str], str],
    ) -> None:
        for c_rate in self.experiments.keys():
            dataframe = self.experiments[c_rate].df
            _, num_columns = dataframe.shape

            time_start = self.csv_formatter.data_start_column

            for time_column in tqdm(range(time_start, num_columns)):
                # Get the timestep from the column header
                time = self.csv_formatter.get_timestep(
                    dataframe,
                    time_column,
                )

                # Get the solmap array
                im = self.create_solmap_image(
                    c_rate,
                    time_column,
                )

                output_dir = os.path.join(
                    self.micro_path, "col/"
                )

                try:
                    os.mkdir(output_dir)
                except FileExistsError:
                    pass

                filename = fname_fn(c_rate, time)
                filename = os.path.join(output_dir, filename)
                save_numpy_arr(im, filename)

    def create_solmap_image(
        self,
        c_rate: str,
        time_column: int,
    ) -> np.ndarray:
        if c_rate not in self.experiments:
            raise KeyError(
                "Provided discharge c-rate {} is not in ".format(c_rate) +
                "the list of experiments"
            )

        # Which discharge we're on
        experiment = self.experiments[c_rate]

        df = experiment.df
        _, num_columns = df.shape

        if time_column >= num_columns:
            raise IndexError(
                "Time column ({}) provided exceeds ".format(time_column) +
                "columns stored in dataframe ({})".format(num_columns)
            )

        # Create a blank array to house State-of-Lithiation data from CSV files
        im = np.zeros((
            self.h_cell * self.scale,
            self.L * self.scale,
            1))

        # Get SoL values located on mesh vertices from COMSOL. We will do an
        # interpolation using these values.
        avail_sol = df.iloc[:, time_column].to_numpy()

        im[
            experiment.Y,
            experiment.X,
            :,
        ] = avail_sol.reshape((avail_sol.size, 1))

        for idx in tqdm(range(len(self.particles))):
            circle = self.particles[idx]

            # Get interpolated coordinates and values
            (
                x_inter,
                y_inter,
                sol_inter
            ) = self.interpolate_circle_sol(
                circle,
                time_column,
                df,
                experiment.x_col,
                experiment.y_col,
            )

            # Fill in interpolated SoL values
            im[
                y_inter,
                x_inter,
                :
            ] = sol_inter.reshape((sol_inter.size, 1))

            im[~self.electrode_mask, :] = [0]

        return im

    def interpolate_circle_sol(
            self,
            circle: typings.Circle_Info,
            time_column: int,
            df: pd.DataFrame,
            x_df_column: pd.DataFrame,
            y_df_column: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r''' Interpolates the State-of-Lithiation using the values returned in
        the COMSOL vertices.
        '''

        to_pix = self.m_to_um * self.scale

        x, y, R = circle["x"], circle["y"], circle["R"]

        xx, yy = get_inscribing_meshgrid(x, y, R, self.grid_size)
        xx, yy = get_coords_in_circle(x, y, R, (xx, yy))

        # (x, y) values within the circle with values to interpolate from
        (
            x_vertices,
            y_vertices,
            SoL_vertices,
        ) = self._get_filled_vertices(
            df,
            time_column,
            x, y, R,
            x_df_column,
            y_df_column,
        )

        interpolant_coords = np.array([x_vertices, y_vertices])
        interpolant_coords = interpolant_coords.T

        # Interpolate missing values
        # COMSOL uses linear interpolation; citation "..."
        SoL_mesh = griddata(
            interpolant_coords,
            SoL_vertices,
            (xx, yy),
            method='linear'
        )

        x_inter, y_inter, sol_inter = self._drop_nan_from_interpolate(
            x_df_column,
            y_df_column,
            (xx, yy),
            SoL_mesh,
        )

        # Convert from micrometer- to pixel_scale
        x_inter = np.ceil(x_inter * to_pix).astype(np.int64) - 1
        y_inter = np.ceil(y_inter * to_pix).astype(np.int64) - 1

        # return (xx, yy, SoL_mesh)
        return (x_inter, y_inter, sol_inter)

    def _get_filled_vertices(
        self,
        df: pd.DataFrame,
        time_column: int,
        x: str,
        y: str,
        R: str,
        x_df_column: pd.DataFrame,
        y_df_column: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Vertices returned from COMSOL
        x_min, x_max, y_min, y_max = get_inscribing_coords(x, y, R)

        filtered_by_x = df[df[x_df_column].between(x_min, x_max)]
        filtered_by_y = filtered_by_x[
            filtered_by_x[y_df_column].between(y_min, y_max)
        ]

        # Prefilled coordinates for the current circle
        x_vertices = filtered_by_y[x_df_column].to_numpy()
        y_vertices = filtered_by_y[y_df_column].to_numpy()
        SoL_vertices = filtered_by_y.iloc[
            :,
            time_column,
        ].to_numpy()

        return (x_vertices, y_vertices, SoL_vertices)

    def _drop_nan_from_interpolate(
        self,
        x_df_column: pd.DataFrame,
        y_df_column: pd.DataFrame,
        meshgrid: typings.meshgrid,
        SoL_interpolated: typings.meshgrid
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # This drops all NaNs in the same location
        xx, yy = meshgrid

        to_drop_df = pd.DataFrame({
            x_df_column: xx,
            y_df_column: yy,
            "temp": SoL_interpolated,
        })

        to_drop_df = to_drop_df.dropna()

        x_filt = to_drop_df[x_df_column].to_numpy()
        y_filt = to_drop_df[y_df_column].to_numpy()
        sol_filt = to_drop_df.iloc[:, 2].to_numpy()

        return (x_filt, y_filt, sol_filt)
