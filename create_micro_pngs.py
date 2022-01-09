from typing import Tuple

import numpy as np

from utils.io import load_json
from utils.io import save_micro_png

from utils.numerics import get_coords_in_circle
from utils.numerics import get_inscribing_meshgrid

RBG_DIM = 3

GREEN = np.array([0, 128, 0])
TEAL = np.array([0, 128, 128])

MESHGRID = Tuple[np.array, np.array]


def create_micro_png(
    micro_hash: dict[str],
        L: int,
        h_cell: int,
        grid_size: int,
        scale: int,
) -> np.array:
    circ_list = micro_hash["circles"]

    micro_im = np.zeros(shape=(h_cell * scale, L * scale, RBG_DIM),
                        dtype=int)

    for circ in circ_list:
        x, y, R = circ["x"], circ["y"], circ["R"]

        xx, yy = get_inscribing_meshgrid(x, y, R, grid_size)
        xx, yy = get_coords_in_circle(x, y, R, (xx, yy))

        micro_im = fill_circle_with_colour(micro_im,
                                           (xx, yy),
                                           scale)

    pore_space = np.all(micro_im == [0, 0, 0], axis=-1)
    micro_im[pore_space] = GREEN

    return micro_im


def fill_circle_with_colour(micro_im: np.array,
                            meshgrid: MESHGRID,
                            scale: int) -> np.array:
    _to_pix = 1e6 * scale

    xx, yy = meshgrid
    xx = np.copy(xx)
    yy = np.copy(yy)

    xx = np.ceil(xx * _to_pix).astype("int") - 1
    yy = np.ceil(yy * _to_pix).astype("int") - 1

    micro_im[yy, xx, :] = TEAL

    return micro_im


if __name__ == "__main__":
    settings = load_json("specs.json")
    ret = load_json("metadata.json")
    for i, micro in enumerate(ret):
        micro_im = create_micro_png(
            micro,
            settings["L"],
            settings["h_cell"],
            settings["grid_size"],
            settings["scale"]
        )
        save_micro_png(
            micro_im,
            "micro_" + str(i + 1) + ".png"
        )
