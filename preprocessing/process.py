from typing import List, Dict, Callable, Tuple

from utils import typings

from utils.io import save_numpy_arr

from utils.numerics import get_inscribing_meshgrid
from utils.numerics import get_inscribing_coords
from utils.numerics import get_coords_in_circle

from preprocessing.data import Experiment
from preprocessing.data import COMSOL_Electrochem_CSV

import os
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.interpolate import griddata


class Microstructure_Backdoor():

    def __init__(
        self,
        csv_formatter: COMSOL_Electrochem_CSV,
        micro_path: str,
        microstructure: np.ndarray,
        L: int,
        h_cell: int,
        c_rate: str,
        time: str,
        particles: List[typings.Circle_Info],
        grid_size: int = 1000,
        scale: int = 10,
    ):
        self.csv_formatter = csv_formatter
        self.micro_path = micro_path

        self.microstructure = microstructure

        self.L = L
        self.h_cell = h_cell
        self.particles = particles

        self.c_rate = c_rate
        self.time = time

        self.grid_size = grid_size
        self.scale = scale

        # Get experiment (csv file of micro under `c-rate` discharge)
        self.experiment = self._get_experiment()
        self.time_column = self._get_time_col()

    def _get_experiment(self) -> Dict[str, Experiment]:
        dataframe = self.csv_formatter.read_csv_into_pandas(
            self.micro_path,
            self.c_rate,
        )
        experiment = Experiment(
            self.c_rate,
            dataframe,
            scale=self.scale,
        )
        return {self.c_rate: experiment}

    def _get_time_col(self):

        dataframe = self.experiment[self.c_rate].df
        _, num_columns = dataframe.shape
        time_start = self.csv_formatter.data_start_column

        time_columns = [col_num for col_num in range(time_start, num_columns)]
        times_in_csv = [self.csv_formatter.get_timestep(
            dataframe, col_num) for col_num in time_columns]

        time_match = [self.time == time for time in times_in_csv]

        idx = 0
        for i in range(len(time_match)):
            # If the time column matches the specified time, then return the
            # corresponding column number in the dataframe
            if time_match[idx]:
                return time_columns[idx]

            idx += 1

    def get_solmap(self) -> np.ndarray:
        electrode_mask = np.ones(
            (self.h_cell * self.scale, self.L * self.scale),
            dtype=bool,
        )

        # Get the solmap array
        solmap = _Interpolate_FEM_Data.create_solmap_image(
            self.experiment,
            self.c_rate,
            self.time_column,
            electrode_mask,
            self.particles,
            self.L,
            self.h_cell,
            self.scale,
            self.grid_size,
        )

        return solmap


class Microstructure():

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
        self.electrode_mask = np.ones(
            (self.h_cell * self.scale, self.L * self.scale),
            dtype=bool,
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
                im = _Interpolate_FEM_Data.create_solmap_image(
                    self.experiments,
                    c_rate,
                    time_column,
                    self.electrode_mask,
                    self.particles,
                    self.L,
                    self.h_cell,
                    self.scale,
                    self.grid_size,
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


class _Interpolate_FEM_Data():

    r'''_Interpolate_FEM_Data are static methods which are the main processing
    steps to create a voxelization of the State-of-Lithiation maps from COMSOL
    simulations.

    A voxelized format of the FEM data is created at a particular C-rate and
    time for a particular microstructure in `create_solmap_image`. In this
    function, the SoL is interpolated for each particle from the associated SoL
    values from the vertices in the FEM mesh.

    Interpolation of SoL for each particle is handled by
    `interpolate_circle_sol`.
    '''

    m_to_um: float = 1e6

    @staticmethod
    def create_solmap_image(
        experiments: Dict[str, Experiment],
        c_rate: str,
        time_column: int,
        electrode_mask: np.ndarray,
        particles: List[typings.Circle_Info],
        L: int,
        h_cell: int,
        scale: int,
        grid_size: int,
    ) -> np.ndarray:

        if c_rate not in experiments:
            raise KeyError(
                "Provided discharge c-rate {} is not in ".format(c_rate) +
                "the list of experiments"
            )

        # `experiment` represents the FEM simulation data at the specified
        # `c_rate`
        experiment = experiments[c_rate]

        df = experiment.df
        _, num_columns = df.shape

        if time_column >= num_columns:
            raise IndexError(
                "Time column ({}) provided exceeds ".format(time_column) +
                "columns stored in dataframe ({})".format(num_columns)
            )

        # Create a blank array to house State-of-Lithiation data from CSV files
        im = np.zeros((
            h_cell * scale,
            L * scale,
            1))

        # Get SoL values located on mesh vertices from COMSOL. We will do an
        # interpolation using these values.
        avail_sol = df.iloc[:, time_column].to_numpy()

        im[
            experiment.Y,
            experiment.X,
            :,
        ] = avail_sol.reshape((avail_sol.size, 1))

        for idx in tqdm(range(len(particles))):
            circle = particles[idx]

            # Get interpolated coordinates and values
            (
                x_inter,
                y_inter,
                sol_inter
            ) = _Interpolate_FEM_Data.interpolate_circle_sol(
                circle,
                time_column,
                df,
                experiment.x_col,
                experiment.y_col,
                scale,
                grid_size,
            )

            # Fill in interpolated SoL values
            im[
                y_inter,
                x_inter,
                :
            ] = sol_inter.reshape((sol_inter.size, 1))

            im[~electrode_mask, :] = [0]

        return im

    @staticmethod
    def interpolate_circle_sol(
        circle: typings.Circle_Info,
        time_column: int,
        df: pd.DataFrame,
        x_df_column: pd.DataFrame,
        y_df_column: pd.DataFrame,
        scale: int,
        grid_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        r''' `interpolate_circle_sol` interpolates the State-of-Lithiation using
        the values returned in the COMSOL vertices.
        '''

        to_pix = _Interpolate_FEM_Data.m_to_um * scale

        x, y, R = circle["x"], circle["y"], circle["R"]

        xx, yy = get_inscribing_meshgrid(x, y, R, grid_size)
        xx, yy = get_coords_in_circle(x, y, R, (xx, yy))

        # (x, y) values within the circle with values to interpolate from
        (
            x_vertices,
            y_vertices,
            SoL_vertices,
        ) = _Interpolation_Helpers._get_filled_vertices(
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

        (x_inter, y_inter,
         sol_inter) = _Interpolation_Helpers._drop_nan_from_interpolate(
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


class _Interpolation_Helpers():

    r''' `_Interpolation_Helpers` has two helper functions to deal with SoL
    values from FEM Simulation.

    1. `_get_filled_vertices` returns the associated FEM verticles within a
        particle with SoL values, as tabulated in a CSV file containing the
        simulation data.
    2. `_drop_nan_from_interpolate` drops all `NaN` values after interpolating
        the SoL values from the mesh verticles to obtain SoL values for the
        whole particle.
    '''

    @staticmethod
    def _get_filled_vertices(
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

    @staticmethod
    def _drop_nan_from_interpolate(
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
