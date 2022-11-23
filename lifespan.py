from create_micro_pngs import create_micro_png
from preprocessing.data import COMSOL_Electrochem_CSV
from preprocessing.process import Microstructure_Backdoor

from etl.extract import _Extraction_Functionality, SOLmap

from utils.io import load_json
from utils.image import get_electrode_meshgrid, zoom_image

import numpy as np
from math import ceil

from utils.typings import Microstructure_Data, Circle_Info, meshgrid


def lifespan(
    microstructure: np.ndarray,
    solmap: np.ndarray,
    particle_id,
    width_wrt_radius: float,
    output_img_size: int,
    params,
    order_zoom: int = 0,
):
    ##############################################
    # CREATE VOXELIZED MICROSTRUCTURE AND SOLMAP #
    ##############################################

    # Load microstructure information from the `json` file
    micro_json = load_json(params["filename"])
    micro_info = micro_json[params["micro_id"]]

    if microstructure is None or solmap is None:
        microstructure, solmap = get_voxelized_arrays(
            micro_info,
            params["micro_path"],
            params["c_rate"],
            params["time"],
        )

    ###################################################################
    # EXTRACT INPUT AND OUTPUT ARRAYS FROM VOXELIZED MICRO AND SOLMAP #
    ###################################################################

    # Create a meshgrid across the electrode
    meshgrid = get_electrode_meshgrid(
        int(micro_info["length"]),
        h=100,
        scale=5,
    )

    particle = micro_info["circles"][particle_id]
    input_im, output_im = extract_particle(
        particle,
        width_wrt_radius,
        scale=5,
        microstructure=microstructure,
        solmap=SOLmap("", solmap),
        meshgrid=meshgrid,
    )

    ##########################################
    # ZOOM EXTRACTED IMAGE TO SPECIFIED SIZE #
    ##########################################

    z_input_im, zoom_factor = zoom_image(input_im, output_img_size,
                                         order=order_zoom)
    z_output_im, _ = zoom_image(output_im, output_img_size,
                                order=order_zoom)

    ##############################
    # "UNZOOMED" EXTRACTED IMAGE #
    ##############################

    zoomed_img_size, _, _ = z_input_im.shape
    uz_input_im, _ = zoom_image(z_input_im,
                                zoomed_img_size / zoom_factor,
                                order=order_zoom)
    uz_output_im, _ = zoom_image(z_output_im,
                                 zoomed_img_size / zoom_factor,
                                 order=order_zoom)

    orig_arrs = (microstructure, solmap)
    extracted_arrs = (input_im, output_im)
    zoomed_extracted = (z_input_im, z_output_im)
    unzoomed_extracted = (uz_input_im, uz_output_im)

    return (orig_arrs, extracted_arrs, zoomed_extracted, unzoomed_extracted)


def extract_particle(
    particle: Circle_Info,
    width_wrt_radius: float,
    scale: int,
    microstructure: np.ndarray,
    solmap: np.ndarray,
    meshgrid: meshgrid,
):
    R = particle["R"]
    # Size image according to radius factor
    box_radius = ceil(float(R) * scale * width_wrt_radius)

    (input_im,
     label_im) = _Extraction_Functionality.extract_input_and_cmap_im(
        box_radius,
        box_radius,
        microstructure,
        solmap,
        particle,
        meshgrid,
        padding_encoding=0,
        scale=5,
        for_training=True,
    )

    return input_im, label_im


def get_voxelized_arrays(
    micro_info: Microstructure_Data,
    micro_path: str,
    c_rate: str,
    time: str,

):
    # Create the "voxelized microstructure image"
    microstructure = create_micro_png(
        micro_info,
        h_cell=100,
        grid_size=1000,
        scale=5,
        pore_phase=65535,
        solid_phase=30000,
    )

    # Create the "voxelized solmap image"
    echem_csv_formatter = COMSOL_Electrochem_CSV()
    backdoor = Microstructure_Backdoor(
        echem_csv_formatter,
        micro_path,
        microstructure,
        L=int(micro_info["length"]),
        h_cell=100,
        c_rate=c_rate,
        time=time,
        particles=micro_info["circles"],
        grid_size=1000,
        scale=5,
    )
    solmap = backdoor.get_solmap()

    return microstructure, solmap


if __name__ == "__main__":
    pass
