from typing import List

from utils.io import load_json
from utils import typings

from preprocessing.data import COMSOL_Electrochem_CSV
from preprocessing.process import Microstructure


if __name__ == "__main__":
    settings = load_json("colourmap_specs.json")

    L = settings["L"]
    h_cell = settings["h_cell"]
    c_rates = settings["c_rates"]

    grid_size = settings["grid_size"]
    scale = settings["scale"]

    to_parse_out = settings["substrings_to_parse_out"]

    # Handles all things electrochem csv!
    echem_csv_handler = COMSOL_Electrochem_CSV()
    microstructure_data: List[typings.Microstructure_Data] = load_json(
        "metadata.json")

    for idx, data in enumerate(microstructure_data):
        micro = Microstructure(
            echem_csv_handler,
            micro_path=str(idx + 1),
            L=L,
            h_cell=h_cell,
            c_rates=c_rates,
            particles=data["circles"],
            grid_size=grid_size,
            scale=scale,
        )

        micro.create_and_save_all_colormaps_from_experiments(
            lambda c_rate, time: "c%s_t%s.npy" % (c_rate, time)
        )
