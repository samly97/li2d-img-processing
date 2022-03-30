import numpy as np

from utils.io import load_json
from utils.io import save_numpy_arr

from utils.numerics import get_coords_in_circle
from utils.numerics import get_inscribing_meshgrid

from utils import typings


def create_micro_png(
    micro_hash: typings.Microstructure_Data,
    h_cell: int,
    grid_size: int,
    scale: int,
    pore_phase: int = 2 ** 16 - 1,
    solid_phase: int = 2 ** 16 - 2,
) -> np.ndarray:
    circ_list = micro_hash["circles"]
    L = int(micro_hash["length"])

    micro_im = np.zeros(shape=(h_cell * scale, L * scale, 1),
                        dtype=np.uint16)

    for circ in circ_list:
        x, y, R = circ["x"], circ["y"], circ["R"]

        xx, yy = get_inscribing_meshgrid(x, y, R, grid_size)
        xx, yy = get_coords_in_circle(x, y, R, (xx, yy))

        micro_im = fill_circle_with_colour(
            micro_im,
            (xx, yy),
            scale,
            solid_phase,
        )

    pore_space = np.all(micro_im == [0], axis=-1)
    micro_im[pore_space] = pore_phase

    return micro_im


def fill_circle_with_colour(
    micro_im: np.ndarray,
    meshgrid: typings.meshgrid,
    scale: int,
    solid_phase: int = 2 ** 16 - 2
) -> np.ndarray:
    _to_pix = 1e6 * scale

    xx, yy = meshgrid
    xx = np.copy(xx)
    yy = np.copy(yy)

    xx = np.ceil(xx * _to_pix).astype(np.int32) - 1
    yy = np.ceil(yy * _to_pix).astype(np.int32) - 1

    micro_im[yy, xx, :] = solid_phase

    return micro_im


if __name__ == "__main__":
    settings = load_json("specs.json")
    ret = load_json("metadata.json")
    for i, micro in enumerate(ret):
        micro_im = create_micro_png(
            micro,
            settings["h_cell"],
            settings["grid_size"],
            settings["scale"],
            pore_phase=settings["pore_phase"],
            solid_phase=settings["solid_phase"],
        )
        save_numpy_arr(
            micro_im,
            "micro_" + str(i + 1) + ".npy"
        )
