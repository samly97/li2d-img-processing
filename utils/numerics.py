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
    r'''
    Inputs:
    - x: (x,) of particle center in micrometers
    - y: (,y) of particle center in micrometers
    - R: radius of ceneter in micrometers
    - grid_size: number of points in the meshgrid

    Returns:
    - (xx, yy): meshgrid of a square box region encapsulating where the
        particle of interest is.
    '''

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
    r''' get_inscribing_coords returns a  tuple indicating the (min, max)
    values of where the particle "ends" essentially. Used to construct a
    bounding meshgrid around the particle of interest.
    '''

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
    r''' get_coords_in_circle determines the indexes in the meshgrid which are
    in the particle of interest. This method is used to determine which pixels
    to fill with colour.
    '''

    xx, yy = meshgrid
    xx = np.copy(xx)
    yy = np.copy(yy)

    x, y, R = float(x) * to_um, float(y) * to_um, float(R) * to_um

    in_circ = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) <= R

    xx = xx[in_circ]
    yy = yy[in_circ]

    return (xx, yy)
