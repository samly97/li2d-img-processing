from typing import Dict, List, Tuple

import json
import os

import numpy as np

from utils import typings
from utils import load_json
from etl.extract import Microstructure_Breaker_Upper


def create_output_dirs(
    input_dir: str,
    label_dir: str,
) -> Tuple[str, str, str]:
    curr_dir = os.getcwd()
    dataset_dir = os.path.join(curr_dir, "dataset")
    input_image_dir = os.path.join(dataset_dir, input_dir)
    label_image_dir = os.path.join(dataset_dir, label_dir)

    _create_dir(dataset_dir)
    _create_dir(input_image_dir)
    _create_dir(label_image_dir)

    return (dataset_dir,
            input_image_dir,
            label_image_dir)


def _create_dir(dir: str) -> None:
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


if __name__ == "__main__":
    # Load user settings
    settings = load_json("dataset_specs.json")

    # Create directories and return path of each
    (
        dataset_dir,
        input_dir,
        label_dir,
    ) = create_output_dirs(
        settings["input_dir"],
        settings["label_dir"],
    )

    # Load metadata generated from COMSOL
    microstructure_data: List[typings.Microstructure_Data] = load_json(
        "metadata.json"
    )

    def micro_arr_fname(num): return "micro_%s.npy" % (num)

    dataset_json: Dict[str, typings.Metadata] = {}

    pic_num = 1

    for idx, data in enumerate(microstructure_data):
        micro = Microstructure_Breaker_Upper(
            micro_num=str(idx + 1),
            solmap_path=os.path.join(
                os.getcwd(),
                str(idx + 1),
            ),
            micro_arr=np.load(micro_arr_fname(str(idx + 1))),
            L=settings["L"],
            h_cell=settings["h_cell"],
            particles=data["circles"],
            scale=settings["scale"],
            sol_max=settings["sol_max"],
            pore_encoding=settings["pore_encoding"],
            padding_encoding=settings["padding_encoding"],
        )

        pic_num, meta_dict = micro.ml_data_from_all_solmaps(
            settings["width_wrt_radius"],
            settings["img_size"],
            input_dir,
            label_dir,
            pic_num,
        )

        dataset_json.update(meta_dict)

    with open(
        os.path.join(dataset_dir, "dataset.json"),
        "w",
    ) as outfile:
        json.dump(dataset_json, outfile, indent=4)
