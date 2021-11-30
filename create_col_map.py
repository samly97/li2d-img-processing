#########################################
## Create COLOUR MAPS from COMSOL Data ##
#########################################

import json
import pandas as pd

import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm

from PIL import Image

import os

from typing import List, Tuple

import matplotlib.pyplot as plt

RGB_DIM = 3

GREEN = np.array([0, 128, 0])

# Wrapper to avoid having to index everytime to discard the
# transparence value when calling the colormap
class nice_cm:
	
	def __init__(self, color: str):
		self.cm = cm.get_cmap(color)

	def __call__(self, val: np.array):
		return self.cm(val)[:, :3]

def get_micro_directories() -> List[str]:
	# Directories where the "electro_c-rate.csv" live
	micro_dirs = os.listdir(os.getcwd())
	# Parse out directories - i.e., no png/json/ipynb files
	micro_dirs = [file for file in micro_dirs 
					if not any(True for substr in ['.png', '.json', '.ipynb', '.DS_Store'] 
					if substr in file)]

	micro_dirs.sort()
	micro_dirs = [os.path.join(os.getcwd(), dirr) for dirr in micro_dirs]

	return micro_dirs

## Can refactor into common utils
def read_user_settings() -> dict:
	f = open("colourmap_specs.json", "r")
	ret = json.load(f)
	return ret

## Can refactor into common utils
def read_metadata(micro_fname: str) -> list:
	f = open(micro_fname, "r")
	ret = json.load(f)
	return ret

# Formats the electrochem filename
def get_electrochem_csv(c_rate: str) -> str:
	fname = "electro_%s.csv" % (c_rate)
	return fname

# For a specific:
# - microstructure
# - c-rate
def read_csv_into_pandas(micro_path: str,
						 c_rate: str,
						 header_row: int) -> pd.DataFrame:
	# Get the electrochem file
	c_rate_csv = get_electrochem_csv(c_rate)
	file_path = os.path.join(micro_path, c_rate_csv)

	dataframe = pd.read_csv(file_path, header=header_row)

	# Drop all NaN values from table
	dataframe = dataframe.dropna()

	# Normalize State-of-Lithiation to 0 -1 scale
	dataframe.iloc[:, 2:] = dataframe.iloc[:, 2:] / 100

	return dataframe

# get_colored_vertices returns a X and Y vector for values that
# are associated with a SoL value from COMSOL
#
# We use these to interpolate
#
# In pixel and not micrometer scale
def get_colored_vertices(df: pd.DataFrame,
						 scale: int) -> Tuple[np.array, np.array]:
	_to_pix = 1e6 * scale

	X_col = dataframe.columns[0]
	Y_col = dataframe.columns[1]

	X = dataframe[X_col].to_numpy()
	Y = dataframe[Y_col].to_numpy()

	# Converting from micrometers to pixel
	X = np.ceil( X * _to_pix).astype(np.int64) - 1
	Y = np.ceil( Y * _to_pix).astype(np.int64) - 1
	
	return (X, Y)

# Gets the SoL for a particular:
# - microstructures
# - time associated with a c_rate
def get_SoL(df: pd.DataFrame, col: int) -> np.array:
	# column represents the timestep
	return df.iloc[:, col]

if __name__ == "__main__":
	settings = read_user_settings()

	L = settings["L"]
	h_cell = settings["h_cell"]
	c_rates = settings["c_rates"]

	scale = settings["scale"]

	micro_dirs = get_micro_directories()
	microstructures = read_metadata("metadata.json")

	## COLOR MAP
	col_map = nice_cm(settings["colormap"])


	for l, micro in enumerate(microstructures):
		# Get the path to the microstructure for file read/write
		micro_path = micro_dirs[l]

		for c_rate_exp in c_rates:
			# Helpful print statement to indicate progress
			print(("Microstructure: %d C-rate: %s") % (l + 1, c_rate_exp))

			dataframe = read_csv_into_pandas(micro_path,
				c_rate_exp,
				settings["header_row"])

			X, Y = get_colored_vertices(dataframe, scale)

			# Get number of columns... 2 - NUM_COLUMN is num of time steps
			_, NUM_COL = dataframe.shape

			for t in range(2, NUM_COL):
				# Create blank image to house SoL data from CSV files
				im = np.zeros((h_cell * scale, L * scale, RGB_DIM))

				# SoL values from the vertices
				avail_sol = get_SoL(dataframe, t)

				# Fill in the image with the available values just to SEE
				im[Y, X, :] = col_map(avail_sol)
				plt.imshow(im)
				plt.show()

				break

			break
		break





