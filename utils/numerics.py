from typing import Tuple
import numpy as np

from . import typings


def get_inscribing_meshgrid(
    x: str,
    y: str,
    R: str,
    grid_size: int,
    to_um: float = 1e-6,
) -> typings.meshgrid:
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
    to_um: float = 1e-6,
) -> Tuple[float, float, float, float]:
    r''' get_inscribing_coords returns a  tuple indicating the (min, max)
    values of where the particle "ends" essentially. Used to construct a
    bounding meshgrid around the particle of interest.
    '''

    x_float = float(x)
    y_float = float(y)
    R_float = float(R)

    x_min = (x_float - R_float) * to_um
    x_max = (x_float + R_float) * to_um
    y_min = (y_float - R_float) * to_um
    y_max = (y_float + R_float) * to_um

    return (x_min, x_max, y_min, y_max)


def get_coords_in_circle(
    x: str,
    y: str,
    R: str,
    meshgrid: typings.meshgrid,
    to_um: float = 1e-6,
) -> typings.meshgrid:
    r''' get_coords_in_circle determines the indexes in the meshgrid which are
    in the particle of interest. This method is used to determine which pixels
    to fill with colour.
    '''

    xx, yy = meshgrid
    xx = np.copy(xx)
    yy = np.copy(yy)

    x_float = float(x) * to_um
    y_float = float(y) * to_um
    R_float = float(R) * to_um

    in_circ = np.sqrt((xx - x_float) ** 2 + (yy - y_float) ** 2) <= R_float

    xx = xx[in_circ]
    yy = yy[in_circ]

    return (xx, yy)


def get_electrode_meshgrid(L: int,
                           h: int,
                           scale=10) -> typings.meshgrid:
    r'''
    This function takes in the geometry of the electrode being analyzed and
    returns a tuple of (xx, yy) meshgrid.

    L: length of electrode (um)
    h: height/width of electrode (um)
    scale: scale of colmap_image RELATIVE to the original electrode geometry
        - e.g. if L = 176, h = 100, and scale = 10, then (1760, 1000)
    '''

    # Meshgrid of coordinates of electrode domain
    x_linspace = np.linspace(0, L, num=L * scale)
    y_linspace = np.linspace(0, h, num=h * scale)

    xx, yy = np.meshgrid(x_linspace, y_linspace)

    return (xx, yy)
