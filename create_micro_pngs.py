from typing import List

from utils.io import load_json
from utils.io import save_numpy_arr

from preprocessing.data import COMSOL_Electrochem_CSV
from preprocessing.process import Microstructure
from preprocessing.process import _Interpolate_FEM_Data

from utils import typings

import numpy as np
import os

if __name__ == "__main__":
    settings = load_json("specs.json")

    # Assume that a FEM file with 0.25C is always exists to create a
    # microstructure with
    c_rate = "0.25"

    # The time column starts is assumed to start at column idx 2 in the FEM
    # simulation file
    TIME_START_COL = 2

    echem_csv_handler = COMSOL_Electrochem_CSV()
    microstructure_data: List[typings.Microstructure_Data] = load_json(
        "ml-dataset-raw/metadata.json")

    for idx, data in enumerate(microstructure_data):
        micro = Microstructure(
            echem_csv_handler,
            micro_path=os.path.join("ml-dataset-raw", str(idx + 1)),
            L=int(microstructure_data[idx]["length"]),
            h_cell=settings["h_cell"],
            c_rates=[c_rate],
            particles=data["circles"],
            grid_size=settings["grid_size"],
            scale=settings["scale"],
        )

        interpolated_micro = _Interpolate_FEM_Data.create_solmap_image(
            micro.experiments,
            c_rate,
            TIME_START_COL,
            micro.electrode_mask,
            micro.particles,
            micro.L,
            micro.h_cell,
            micro.scale,
            micro.grid_size,
        )

        solid_phase = interpolated_micro > 0.0
        pore_phase = ~solid_phase

        save_micro = np.zeros_like(interpolated_micro, dtype=np.uint16)
        save_micro[solid_phase] = settings["solid_phase"]
        save_micro[pore_phase] = settings["pore_phase"]

        save_numpy_arr(
            save_micro,
            os.path.join("ml-dataset-raw", "micro_" + str(idx + 1) + ".npy")
        )
