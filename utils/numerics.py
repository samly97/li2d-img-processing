from typing import Tuple
import numpy as np

MESHGRID = Tuple[np.array, np.array]


def get_inscribing_meshgrid(
    x: str,
    y: str,
    R: str,
    grid_size: int,
    to_um: int = 1e-6,
) -> MESHGRID:

    x_min, x_max, y_min, y_max = get_inscribing_coords(x, y, R, to_um)

    x_linspace = np.linspace(x_min, x_max, grid_size)
    y_linspace = np.linspace(y_min, y_max, grid_size)

    xx, yy = np.meshgrid(x_linspace, y_linspace)

    return (xx, yy)


def get_inscribing_coords(
    x: str,
    y: str,
    R: str,
    to_um: int = 1e-6,
) -> Tuple[float, float, float, float]:

    x, y, R = float(x), float(y), float(R)

    x_min = (x - R) * to_um
    x_max = (x + R) * to_um
    y_min = (y - R) * to_um
    y_max = (y + R) * to_um

    return (x_min, x_max, y_min, y_max)


def get_coords_in_circle(
    x: str,
    y: str,
    R: str,
    meshgrid: MESHGRID,
    to_um: int = 1e-6,
) -> MESHGRID:

    xx, yy = meshgrid
    xx = np.copy(xx)
    yy = np.copy(yy)

    x, y, R = float(x) * to_um, float(y) * to_um, float(R) * to_um

    in_circ = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) <= R

    xx = xx[in_circ]
    yy = yy[in_circ]

    return (xx, yy)
