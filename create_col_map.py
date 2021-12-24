import os
from typing import List, Tuple

from tqdm import tqdm

import json
import pandas as pd

import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm

from PIL import Image

RGB_DIM = 3

GREEN = np.array([0, 128, 0])

MESHGRID = Tuple[np.array, np.array]


class nice_cm:
    # Wrapper to avoid having to index everytime to discard the
    # transparence value when calling the colormap

    def __init__(self, color: str):
        self.cm = cm.get_cmap(color)

    def __call__(self, val: np.array):
        return self.cm(val)[:, :3]


def get_micro_directories() -> List[str]:
    # Directories where the "electro_c-rate.csv" live
    micro_dirs = os.listdir(os.getcwd())
    # Parse out directories - i.e., no png/json/ipynb files
    micro_dirs = [
        file for file in micro_dirs
        if not any(True for substr in
                   ['.png', '.json', '.ipynb', '.DS_Store',
                    '.git', '.gitignore', '.py', '_*']
                   if substr in file)
    ]

    micro_dirs.sort()
    micro_dirs = [os.path.join(os.getcwd(), dirr) for dirr in micro_dirs]

    return micro_dirs


def read_user_settings() -> dict:
    # Can refactor into common utils
    f = open("colourmap_specs.json", "r")
    ret = json.load(f)
    return ret


def read_metadata(micro_fname: str) -> list:
    # Can refactor into common utils
    f = open(micro_fname, "r")
    ret = json.load(f)
    return ret


def get_electrochem_csv(c_rate: str) -> str:
    # Formats the electrochem filename
    fname = "electro_%s.csv" % (c_rate)
    return fname


def read_csv_into_pandas(
        micro_path: str,
        c_rate: str,
        header_row: int) -> pd.DataFrame:
    # For a specific:
    # - microstructure
    # - c-rate

    # Get the electrochem file
    c_rate_csv = get_electrochem_csv(c_rate)
    file_path = os.path.join(micro_path, c_rate_csv)

    dataframe = pd.read_csv(file_path, header=header_row)

    # Drop all NaN values from table
    dataframe = dataframe.dropna()

    # Normalize State-of-Lithiation to 0 -1 scale
    dataframe.iloc[:, 2:] = dataframe.iloc[:, 2:] / 100

    return dataframe


def get_colored_vertices(
        df: pd.DataFrame,
        xy_col: Tuple[pd.DataFrame, pd.DataFrame],
        scale: int) -> Tuple[np.array, np.array]:
    # get_colored_vertices returns a X and Y vector for values that
    # are associated with a SoL value from COMSOL
    #
    # We use these to interpolate
    #
    # In pixel and not micrometer scale

    _to_pix = 1e6 * scale

    X_col, Y_col = xy_col

    X = df[X_col].to_numpy()
    Y = df[Y_col].to_numpy()

    # Converting from micrometers to pixel
    X = np.ceil(X * _to_pix).astype(np.int64) - 1
    Y = np.ceil(Y * _to_pix).astype(np.int64) - 1

    return (X, Y)


def _get_timestep(df: pd.DataFrame, col: int) -> str:
    # Get the timestep, t, from the column header
    time = df.columns[col].split(" ")
    time = time[-1].split("=")[-1]
    return time


def create_colormap_image(
        df: pd.DataFrame,
        colormap: nice_cm,
        circles: list[dict[str]],
        time_col: int,
        xy_col: Tuple[pd.DataFrame, pd.DataFrame],
        XY_vertices: Tuple[np.array, np.array],
        h_cell: int,
        L: int,
        grid_size,
        scale: int) -> np.array:
    # create_colormap_image collates everything into one

    # Create blank image to house SoL data from CSV files
    im = np.zeros((h_cell * scale, L * scale, RGB_DIM))

    # Header names of the (x, y) columns in Pandas DataFrame
    X_col, Y_col = xy_col
    # Coordinates already associated with colour (pixel-scale)
    X, Y = XY_vertices

    # Get SoL associated with (X, Y)
    avail_sol = df.iloc[:, time_col].to_numpy()

    # Setting precolored locations
    im[Y, X, :] = colormap(avail_sol)

    for idx in tqdm(range(len(circles))):
        circle = circles[idx]

        # Get interpolated coordinates and values
        x_inter, y_inter, sol_inter = interpolate_circle_color(
            circle,
            time_col,
            df,
            (X_col, Y_col),
            grid_size,
            scale)

        # Fill in color
        im[y_inter, x_inter, :] = colormap(sol_inter)

    return im


def interpolate_circle_color(
        circle: dict[str],
        time_col: int,
        df: pd.DataFrame,
        xy_col: Tuple[pd.DataFrame, pd.DataFrame],
        grid_size: int,
        scale: int) -> Tuple[np.array, np.array, np.array]:
    # Interpolates the State-of-Lithiation, which is represented by the
    # color, using the available SoL values as per the COMSOL vertices.

    _to_pix = 1e6 * scale

    x, y, R = circle["x"], circle["y"], circle["R"]

    xx, yy = _get_inscribing_meshgrid(x, y, R, grid_size)
    xx, yy = _get_coords_in_circle(x, y, R, (xx, yy))

    # (x, y) values within the circle with values to interpolate from
    (
        x_vertices,
        y_vertices,
        SoL_vertices,
    ) = _get_filled_vertices(df,
                             time_col,
                             x, y, R,
                             xy_col)

    interpolant_coords = np.array([x_vertices, y_vertices])
    interpolant_coords = interpolant_coords.T

    # Interpolate missing values
    # COMSOL uses linear interpolation; citation: "..."
    SoL_mesh = griddata(interpolant_coords,
                        SoL_vertices,
                        (xx, yy),
                        method='linear')

    x_inter, y_inter, sol_inter = _drop_nan_from_interpolate(xy_col,
                                                             (xx, yy),
                                                             SoL_mesh)

    # Convert from micrometer- to pixel-scale
    x_inter = np.ceil(x_inter * _to_pix).astype(np.int64) - 1
    y_inter = np.ceil(y_inter * _to_pix).astype(np.int64) - 1

    # return (xx, yy, SoL_mesh)
    return (x_inter, y_inter, sol_inter)


def _drop_nan_from_interpolate(
        xy_col: Tuple[pd.DataFrame, pd.DataFrame],
        meshgrid: MESHGRID,
        SoL_interpolated: MESHGRID) -> Tuple[np.array, np.array, np.array]:
    # This drops all NaNs in the same location

    X_col, Y_col = xy_col
    xx, yy = meshgrid

    to_drop_df = pd.DataFrame({
        X_col: xx,
        Y_col: yy,
        "temp": SoL_interpolated,
    })

    to_drop_df = to_drop_df.dropna()

    x_filt = to_drop_df[X_col].to_numpy()
    y_filt = to_drop_df[Y_col].to_numpy()
    sol_filt = to_drop_df.iloc[:, 2].to_numpy()

    return (x_filt, y_filt, sol_filt)


def _get_inscribing_coords(x: str,
                           y: str,
                           R: str) -> Tuple[float, float, float, float]:
    # Can refactor into common uitls

    # Conversion factor from um to pixels
    _to_um = 1e-6

    x, y, R = float(x), float(y), float(R)

    x_min = (x - R) * _to_um
    x_max = (x + R) * _to_um
    y_min = (y - R) * _to_um
    y_max = (y + R) * _to_um

    return (x_min, x_max, y_min, y_max)


def _get_filled_vertices(
        df: pd.DataFrame,
        time_col: int,
        x: str,
        y: str,
        R: str,
        xy_col: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[np.array,
                                                            np.array,
                                                            np.array]:
    # Vertices which already has color

    X_col, Y_col = xy_col
    x_min, x_max, y_min, y_max = _get_inscribing_coords(x, y, R)

    filtered_by_x = df[df[X_col].between(x_min, x_max)]
    filtered_by_y = filtered_by_x[filtered_by_x[Y_col].between(y_min, y_max)]

    # Prefilled coordinates for the current circle
    x_vertices = filtered_by_y[X_col].to_numpy()
    y_vertices = filtered_by_y[Y_col].to_numpy()
    SoL_vertices = filtered_by_y.iloc[:, time_col].to_numpy()

    return (x_vertices, y_vertices, SoL_vertices)


def _get_inscribing_meshgrid(
        x: str,
        y: str,
        R: str,
        grid_size: int) -> MESHGRID:
    # Can refactor into common utils
    x_min, x_max, y_min, y_max = _get_inscribing_coords(x, y, R)

    x_linspace = np.linspace(x_min, x_max, grid_size)
    y_linspace = np.linspace(y_min, y_max, grid_size)

    xx, yy = np.meshgrid(x_linspace, y_linspace)

    return (xx, yy)


def _get_coords_in_circle(
        x: str,
        y: str,
        R: str,
        meshgrid: MESHGRID) -> MESHGRID:
    # Can refactor into common utils
    _to_um = 1e-6

    xx, yy = meshgrid
    xx = np.copy(xx)
    yy = np.copy(yy)

    x, y, R = float(x) * _to_um, float(y) * _to_um, float(R) * _to_um

    in_circ = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) <= R

    xx = xx[in_circ]
    yy = yy[in_circ]

    return (xx, yy)


def save_colmap_png(
        micro_im: np.array,
        fname: str):
    # Can refactor into common utils
    im = Image.fromarray((255 * micro_im).astype(np.uint8))
    im.save(fname)


if __name__ == "__main__":
    settings = read_user_settings()

    L = settings["L"]
    h_cell = settings["h_cell"]
    c_rates = settings["c_rates"]

    grid_size = settings["grid_size"]
    scale = settings["scale"]

    micro_dirs = get_micro_directories()
    microstructures = read_metadata("metadata.json")

    # COLOR MAP
    col_map = nice_cm(settings["colormap"])

    for idx, micro in tqdm(enumerate(microstructures)):
        # Get the path to the microstructure for file read/write
        micro_path = micro_dirs[idx]

        for c_rate_exp in c_rates:
            # Helpful print statement to indicate progress
            print(("Microstructure: %d C-rate: %s") % (idx + 1, c_rate_exp))

            dataframe = read_csv_into_pandas(
                micro_path,
                c_rate_exp,
                settings["header_row"])

            X_col = dataframe.columns[0]
            Y_col = dataframe.columns[1]

            X, Y = get_colored_vertices(
                dataframe,
                (X_col, Y_col),
                scale)

            # Get number of columns... 2 - NUM_COLUMN is num of time steps
            _, NUM_COL = dataframe.shape

            for t in tqdm(range(2, NUM_COL)):

                # Get the timestep from the column header
                time = _get_timestep(dataframe, t)

                # Get colormap image
                im = create_colormap_image(
                    dataframe,
                    col_map,
                    micro["circles"],
                    t,
                    (X_col, Y_col),
                    (X, Y),
                    h_cell,
                    L,
                    grid_size,
                    scale)

                output_dir = os.path.join(micro_path, "col/")
                try:
                    os.mkdir(output_dir)
                except FileExistsError:
                    pass

                filename = "c%s_t%s.png" % (c_rate_exp, time)
                filename = os.path.join(output_dir, filename)
                save_colmap_png(im, filename)
