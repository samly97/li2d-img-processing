import os
from typing import Tuple, List
from tqdm import tqdm
import json
import numpy as np
from math import ceil
from skimage import io

from utils.io import load_json
from utils.io import save_micro_png
from utils.image import zoom_image
from utils.image import pad_image
from utils.image import padded_coords
from utils.metrics import measure_porosity

####################################
# CREATE MACHINE LEARNING DATASET ##
####################################

# TODO:
# - ACTIVATIONS: seperate geometry from concentration prediction ✅
# - LOCAL porosity ✅
# - There are some "bleeding" edge effects from the colors not interpolating
#   the colors completely. Will need to decide what to do. Could decide to
#   interpolate the edges or we could use the difference between the circles
#   and the generated color and just fill it in with the background.
# - incorporate BOUNDING_BOX parameterization ✅
# - zoom ratios ✅


GREEN = np.array([0, 128, 0])
TEAL = np.array([0, 128, 128])

# Custom typings
MESHGRID = Tuple[np.array, np.array]


def create_output_dirs(input_dir: str,
                       label_dir: str) -> Tuple[str, str, str]:

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


def _create_dir(dir: str):
    try:
        os.mkdir(dir)
    except FileExistsError:
        pass


def get_electrode_meshgrid(L: int,
                           h: int,
                           scale=10) -> MESHGRID:
    r'''
    This function takes in the geometry of the electrode being analyzed and
    returns a tuple of (xx, yy) meshgrid.

    L: length of electrode (um)
    h: height/width of electrode (um)
    scale: scale of colmap_image RELATIVE to the original electrode geometry
        - e.g. if L = 176, h = 100, and scale = 10, then (1760, 1000)
    '''

    # Meshgrid of coordinates of electrode domain
    x_linspace = np.linspace(0, L, num=L * scale)
    y_linspace = np.linspace(0, h, num=h * scale)

    xx, yy = np.meshgrid(x_linspace, y_linspace)

    return (xx, yy)


def get_exp_params(colmap_fname: str) -> Tuple[str, str]:
    r'''
    Retrieves the simulated C-rate and timestep from the filename.

    Returns: (c_rate, time)
    '''

    # Remove the full path, only the filename in form of:
    # - c(c-rate)_t(time).png
    colmap_fname = colmap_fname.split('/')[-1]

    temp = colmap_fname.split('.png')
    temp = temp[0]
    temp = temp.split('_t')

    # Get C-rate
    temp1 = temp[0]
    c_rate = temp1[1:]

    # Get timestep
    time = temp[1]

    return (c_rate, time)


def get_micro_colmaps(microstructure: int) -> List[str]:
    r'''
    Takes the microstructure being analyzed and being called in the
    same directory where the colour images are being generated. That is,
    being called in the directory where microstructures: 1, 2, 3, etc.; are
    children directories.

    Returns a list of filenames corresponding to the generated colormap images.
    '''
    curr_dir = os.getcwd()

    cm_dir = os.path.join(curr_dir, str(microstructure), "col")
    files = os.listdir(cm_dir)
    colmaps = [os.path.join(cm_dir, colmap) for colmap in files]
    return colmaps


def generate_ml_dataset(
        cell_params: Tuple[int, int],
        settings: Tuple[int, int, int],
        directories: Tuple[str, str, str],
        microstructures,
):
    r'''
    generate_ml_dataset is the top-level function extract the input and target
    images used for training the Machine Learning model. The JSON metadata will
    be generated here as well.
    '''

    L, h_cell = cell_params
    scale, output_img_size, width_wrt_radius = settings
    dataset_dir, input_dir, label_dir = directories

    meshgrid = get_electrode_meshgrid(L, h_cell, scale)

    # Dictionary to write output to
    dataset_json = {}

    # Variable to store current image
    pic_num = 1

    # For each microstructure, e.g. micros [1, 2, 3, 4, 5]
    for idx in tqdm(range(1, len(microstructures) + 1)):
        micro_im = io.imread("micro_" + str(idx) + ".png")

        # Get all colormaps associated with microstructure "i"
        cm_filenames = get_micro_colmaps(idx)

        circles = micro_data[idx - 1]["circles"]

        for colormap in tqdm(cm_filenames):
            cm_image = io.imread(colormap)

            # Get the experimental params from filename
            c_rate, time = get_exp_params(colormap)

            for particle in circles:
                x = particle['x']
                y = particle['y']
                R = particle['R']

                # Size image according to radius factor
                box_radius = ceil(float(R) * scale * width_wrt_radius)

                input_im, label_im = extract_input_and_cmap_im(
                    box_radius,
                    micro_im,
                    cm_image,
                    (x, y, R),
                    meshgrid,
                    scale)

                # Measure local porosity
                porosity = measure_porosity(
                    input_im,
                    GREEN,
                )

                input_im, zoom_factor = zoom_image(input_im, output_img_size)
                label_im, _ = zoom_image(label_im, output_img_size)

                # Save input image
                input_fname = os.path.join(
                    input_dir,
                    str(pic_num) + ".png")
                save_micro_png(input_im, input_fname)

                # Save label image
                label_fname = os.path.join(
                    label_dir,
                    str(pic_num) + ".png")
                save_micro_png(label_im, label_fname)

                # Write metadata to JSON
                dataset_json[pic_num] = {
                    "micro": idx,
                    "x": x,
                    "y": y,
                    "R": R,
                    "zoom_factor": zoom_factor,
                    "c-rate": c_rate,
                    "time": time,
                    "dist_from_sep": float(x)/L,
                    "porosity": porosity,
                }

                # Update picture number
                pic_num += 1

    # Save JSON data
    dataset_json_fname = os.path.join(dataset_dir, "dataset.json")

    with open(dataset_json_fname, 'w') as outfile:
        json.dump(dataset_json, outfile, indent=4)


def extract_input_and_cmap_im(box_radius: int,
                              micro_im: np.array,
                              colmap_im: np.array,
                              circle: Tuple[str, str, str],
                              mesh: MESHGRID,
                              scale: int) -> Tuple[np.array, np.array]:
    r'''
    Gets both the input (blank) and labelled (with color) images
    to form machine learning data.

    Returns: (input_im, labelled_im)
    '''

    x, y, R = circle
    xx, yy = mesh

    # Microstructure image with color in circle of analysis
    with_color = _get_color_circle(
        colmap_im,
        (x, y, R),
        (xx, yy))
    # Apply the particle with color to the microstructure image
    with_color = _add_color_to_background(micro_im, with_color)

    padded_with_color = pad_image(with_color, box_radius)
    padded_micro = pad_image(micro_im, box_radius)

    # New coordinates for (x, y) after padding
    x_new, y_new = padded_coords(
        x,
        y,
        box_radius,
        scale)

    # Extract the "blank" microstructure image and image with particle with
    # color
    input_im = padded_micro[
        y_new - box_radius: y_new + box_radius - 1,
        x_new - box_radius: x_new + box_radius - 1,
        :
    ]
    label_im = padded_with_color[
        y_new - box_radius: y_new + box_radius - 1,
        x_new - box_radius: x_new + box_radius - 1,
        :
    ]

    return input_im, label_im


def _get_color_circle(
        col_map: np.array,
        circle: Tuple[str, str, str],
        mesh: MESHGRID) -> np.array:
    r''' Retrieves the circle from the colormap based on (x, y, R) from
    the COMSOL generated metadata file. "scale" was used to create the
    colormap.
    '''

    col_map = np.copy(col_map)

    x, y, r = circle
    xx, yy = mesh

    x = float(x)
    y = float(y)
    r = float(r)

    # Boolean array where "True" is a pixel inside the circle of analysis
    in_circ = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) <= r

    # "Turn off" all pixels outside of current circle
    ret_im = col_map
    ret_im[~in_circ] = [0, 0, 0]

    return ret_im


def _add_color_to_background(micro_im: np.array,
                             colored_particle_im: np.array) -> np.array:
    r'''Given the background image of the microstructure, this function returns
        a new image which replaces the background of the circle of interest
        with its colormap. Original dimensions of the colormap is preserved.'''

    # Determine the pixels with color
    colored = np.all(colored_particle_im == [0, 0, 0], axis=-1)
    colored = ~colored

    ret_im = np.copy(micro_im)
    # Micro im: remove color from circle of interest
    ret_im[colored] = [0, 0, 0]
    ret_im = ret_im + colored_particle_im

    return ret_im


if __name__ == "__main__":
    # Load user settings
    settings = load_json("dataset_specs.json")

    L = settings["L"]
    h_cell = settings["h_cell"]

    # Resolution of the colormap images relative to the dimensions
    scale = settings["scale"]

    # How big the box should be with respect to the radius of the circle
    width_wrt_radius = settings["width_wrt_radius"]
    output_img_size = settings["img_size"]

    # Create directories and return path of each
    (dataset_dir,
     input_dir,
     label_dir) = create_output_dirs(
        settings["input_dir"],
        settings["label_dir"])

    # Load metadata generated from COMSOL
    micro_data = load_json("metadata.json")

    # Generate Machine Learning data
    generate_ml_dataset(
        (L, h_cell),
        (scale, output_img_size, width_wrt_radius),
        (dataset_dir, input_dir, label_dir),
        micro_data,
    )
