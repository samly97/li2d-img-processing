import os
import json
import numpy as np
from PIL import Image


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


def save_micro_png(
    micro_im: np.array,
    fname: str,
):
    r''' Values to save_micro_png should be [0, 255] for consistency.
    '''
    im = Image.fromarray(micro_im.astype(np.uint8))
    im.save(fname)


def save_numpy_arr(
    micro_im: np.ndarray,
    fname: str,
):
    np.save(fname, micro_im)
