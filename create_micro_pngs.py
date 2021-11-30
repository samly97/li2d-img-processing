#########################################
## GENERATE new microstructural images ##
#########################################

# 1. Read JSON file
# 2. Create a numpy array with lenght (arr_circle)
# 3. For each circle make circle Teal and put into np array
# 4. Sum up all elements in array (reduce) should now have an image that's all coloured circles with black background
# 5. For all pixels where it's black, set to Green

from typing import Tuple

import numpy as np
import json
from PIL import Image

RBG_DIM = 3

GREEN = np.array([0, 128, 0])
TEAL = np.array([0, 128, 128])

MESHGRID = Tuple[np.array, np.array]

def read_user_settings() -> dict:
	f = open("specs.json", "r")
	ret = json.load(f)
	return ret

def read_metadata(micro_fname: str) -> list:
	f = open(micro_fname, "r")
	ret = json.load(f)
	return ret

def save_micro_png(micro_im: np.array,
				   fname: str):
	im = Image.fromarray(micro_im.astype(np.uint8))
	im.save(fname)

def create_micro_png(micro_hash: dict[str],
					 L: int,
					 h_cell: int,
					 grid_size: int,
					 scale: int) -> np.array:
	circ_list = micro_hash["circles"]

	micro_im = np.zeros(shape = (h_cell * scale, L * scale, RBG_DIM), 
							  dtype=int)
	
	for circ in circ_list:
		x, y, R = circ["x"], circ["y"], circ["R"]

		xx, yy = _get_inscribing_meshgrid(x, y, R, grid_size)
		xx, yy = _get_coords_in_circle(x, y, R, (xx, yy))

		micro_im = fill_circle_with_colour(micro_im,
										  (xx, yy),
										   scale)

	pore_space = np.all(micro_im == [0, 0, 0], axis=-1)
	micro_im[pore_space] = GREEN

	return micro_im

def _get_inscribing_coords(x: str,
						   y: str,
						   R: str) -> Tuple[float, float, float, float]:
	# Conversion factor from um to pixels
	_to_um = 1e-6

	x, y, R = float(x), float(y), float(R)

	x_min = (x - R) * _to_um
	x_max = (x + R) * _to_um
	y_min = (y - R) * _to_um
	y_max = (y + R) * _to_um

	return (x_min, x_max, y_min, y_max)

def _get_inscribing_meshgrid(x: str,
						     y: str,
						     R: str,
						     grid_size: int) -> MESHGRID:
	x_min, x_max, y_min, y_max = _get_inscribing_coords(x, y, R)

	x_linspace = np.linspace(x_min, x_max, grid_size)
	y_linspace = np.linspace(y_min, y_max, grid_size)

	xx, yy = np.meshgrid(x_linspace, y_linspace)

	return (xx, yy)

def _get_coords_in_circle(x: str, 
						  y: str, 
						  R: str, 
						  meshgrid: MESHGRID) -> MESHGRID:
	_to_um = 1e-6

	xx, yy = meshgrid
	xx = np.copy(xx)
	yy = np.copy(yy)

	x, y, R = float(x) * _to_um, float(y) * _to_um, float(R) * _to_um

	in_circ = np.sqrt( (xx - x) ** 2 + (yy - y) ** 2) <= R

	xx = xx[in_circ]
	yy = yy[in_circ]

	return (xx, yy)

def fill_circle_with_colour(micro_im: np.array,
							meshgrid: MESHGRID,
							scale: int) -> np.array:
	_to_pix = 1e6 * scale

	xx, yy = meshgrid
	xx = np.copy(xx)
	yy = np.copy(yy)

	xx = np.ceil( xx * _to_pix).astype("int") - 1
	yy = np.ceil( yy * _to_pix).astype("int") - 1

	micro_im[yy, xx, :] = TEAL

	return micro_im

if __name__ == "__main__":
	settings = read_user_settings()
	ret = read_metadata("metadata.json")
	for i, micro in enumerate(ret):
		micro_im = create_micro_png(micro,
			settings["L"],
			settings["h_cell"],
			settings["grid_size"],
			settings["scale"])
		save_micro_png(micro_im,
					   "micro_" + str(i + 1) + ".png")
