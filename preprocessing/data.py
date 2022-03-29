from typing import Callable

import os

import pandas as pd
import numpy as np


class COMSOL_Electrochem_CSV():

    def __init__(
            self,
            format_fn=lambda c_rate: "electro_%s.csv" % (c_rate),
            **kwargs):

        # Settings
        self.header_row: int = kwargs.get("header_row", 8)
        self.data_start_column: int = kwargs.get("data_start_column", 2)

        # Useful functions
        self.format_filename: Callable[[str], str] = format_fn

    def read_csv_into_pandas(
        self,
        micro_path: str,
        c_rate: str,
    ) -> pd.DataFrame:
        c_rate_csv = self.format_filename(c_rate)
        file_path = os.path.join(
            micro_path,
            c_rate_csv
        )

        dataframe = pd.read_csv(
            file_path,
            header=self.header_row,
        )

        # Drop all NaN values from table
        dataframe = dataframe.dropna()

        # Normalize State-of-Lithiation to 0-1 scale
        dataframe.iloc[
            :,
            self.data_start_column:,
        ] = dataframe.iloc[:, self.data_start_column:] / 100

        return dataframe

    def get_timestep(
            self,
            df: pd.DataFrame,
            column: int) -> str:
        # Get the timestep, t, from the column header
        time = df.columns[column].split(" ")
        time = time[-1].split("=")[-1]
        return time


class Experiment():

    m_to_um: float = 1e6

    def __init__(
            self,
            c_rate: str,
            dataframe: pd.DataFrame,
            scale: int = 10,
    ):

        self.c_rate = c_rate
        self.df = dataframe

        self.x_col, self.y_col = self._set_xy_cols()
        self.X, self.Y = self._get_comsol_nodes(scale)

    def _get_comsol_nodes(self, scale: int):
        r''' _get_comsol_nodes returns a X and Y vector for values that are
        associated with a State-of-Lithiation value from COMSOL. We use these
        values to interpolate.
        '''

        # In "pixel scale"
        to_pix = self.m_to_um * scale

        X = self.df[self.x_col].to_numpy()
        Y = self.df[self.y_col].to_numpy()

        # Converting from micrometers to pixel
        X = np.ceil(X * to_pix).astype(np.int64) - 1
        Y = np.ceil(Y * to_pix).astype(np.int64) - 1

        return (X, Y)

    def _set_xy_cols(
        self,
        x_col_idx: int = 0,
        y_col_idx: int = 1,
    ):
        x_col = self.df.columns[x_col_idx]
        y_col = self.df.columns[y_col_idx]
        return (x_col, y_col)
