import os
import json
import numpy as np


def load_json(
    filename: str,
    path: str = "",
):
    if path == "":
        filepath = os.path.join(os.getcwd(), filename)
    else:
        filepath = os.path.join(path, filename)

    f = open(filepath, "r")
    ret = json.load(f)

    return ret


def save_numpy_arr(
    micro_im: np.ndarray,
    fname: str,
):
    np.save(fname, micro_im)
