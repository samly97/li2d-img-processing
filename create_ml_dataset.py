import os
from typing import Tuple, List
from tqdm import tqdm
import json
import numpy as np
from math import ceil
from PIL import Image
from skimage import io
from scipy.ndimage import zoom
import matplotlib.pyplot as plt

import pdb

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

# Current Pseudo Code
"""
1. Loop through the microstructures: [1, 2, ...]

1a. Get dimensions of the microstructural image

1b. From a generic colormap (for a specific microstructural image), get a colormap
    mesh.

    What is this?

    A:

    Literally just a meshgrid of the domain. It doesn't do anything fancy with the
    particle geometry.

1c. Load the microstructure data:
	- List of circles
	- Porosity value (global)
	- Tortuosity value (global)

2. Loop through the generated colormaps [1, 2, ..., 36]

2a. Read the colormap as a NumPy array

2b. Get the experimental parameters:
	- C-rate
	- Timestep, t (seconds)

3. Loop through all particles in the microstructural data

3a. Extract the (input, target) image

3b. Save the input and output images

3c. Write metadata to JSON

4. Write generated metadata to output JSON file

"""

# Thoughts on Pending Modifications/Additions
"""
1. Activation

Depends on:
- Zoom ratio

Can implement this when iterating through each microstructure. For each
microstructure, we can create however many 'just white' circles with whatever
background color we want.

Though when using the Machine Learning model, there needs to be a way to get
the activation back:
- There needs to be a mapping between the IMAGE_NUMBER (e.g. 5000) and its
  activation (e.g. from microstructure 1 circle 55)

2. Local Porosity

Do not necessarily need to use PoreSpy. Can just find out the fraction of the
phases using NumPy and then divide ratios.

Can be incorporated relatively easily within existing implementation.

3. Zoom Ratio


With smaller particles, we'll end up seeing more of the electrode domain, but when
(R/Box_size) is close to 100%, the image will be mostly particle, and doesn't show
the diffusive/migrative path.

How will this be defined? Will it be relative to the:
	(Maximum Radius)/Bounding box size
?

One implementation could be to define a default bounding size, AND THEN zoom
to get a targetted percentage-% of how much pore space we see.

Jeff:
- Something like particle diameter x 2
- Something that's consistent
- But images at the end have to be the same size

"""

GREEN = np.array([0, 128, 0])
TEAL = np.array([0, 128, 128])

# Custom typings
MESHGRID = Tuple[np.array, np.array]


def read_user_settings() -> dict:
    # Can refactor into common utils
    f = open("dataset_specs.json", "r")
    ret = json.load(f)
    return ret


def read_metadata(micro_fname: str) -> list:
    # Can refactor into common utils
    f = open(micro_fname, "r")
    ret = json.load(f)
    return ret


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

                input_im, zoom_factor = _zoom_image(input_im, output_img_size)
                label_im, _ = _zoom_image(label_im, output_img_size)

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

                # Measure local porosity
                porosity = measure_porosity(input_im)

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


def _zoom_image(
    img: np.array,
    output_img_size: int = 200
) -> Tuple[np.array, float]:

    img_size, _, _ = img.shape
    zoom_factor = output_img_size / img_size
    zoom_tuple = (zoom_factor,) * 2 + (1,)

    ret_im = zoom(img, zoom_tuple)
    return ret_im, zoom_factor


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

    padded_with_color = _pad_image(with_color, box_radius)
    padded_micro = _pad_image(micro_im, box_radius)

    # New coordinates for (x, y) after padding
    x_new, y_new = _padded_coords(
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


def measure_porosity(
        micro_im: np.array
) -> float:
    r''' measure_porosity measures the porosity of the extracted images used in
        the machine learning model. These images are different from the
        microstructural image, so this measures the LOCAL (extracted) and not
        the GLOBAL (microstructure) porosity.

        Inputs:
        - micro_im: np.array; extracted image with particle centered

        Returns:
        - porosity: float
    '''

    _black = np.array([0, 0, 0])

    Y, X, _ = micro_im.shape

    padding = np.all(micro_im == _black, axis=-1)
    padding = np.sum(padding)

    pore_space = np.all(micro_im == GREEN, axis=-1)
    pore_space = np.sum(pore_space)

    electrode_domain = float(Y * X - padding)
    porosity = pore_space / electrode_domain

    return porosity


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


def _pad_image(orig_im: np.array,
               pad_width: int,
               pad_type="constant") -> np.array:
    ret_im = np.copy(orig_im)
    ret_im = np.pad(ret_im,
                    (
                        (pad_width, pad_width),
                        (pad_width, pad_width),
                        (0, 0),
                    ), pad_type)

    return ret_im


def _padded_coords(
        x: str,
        y: str,
        pad_width: int,
        scale=10) -> Tuple[int, int]:

    y_new = ceil(float(y) * scale) + pad_width
    x_new = ceil(float(x) * scale) + pad_width

    return (x_new, y_new)


def save_micro_png(
        micro_im: np.array,
        fname: str):
    # Can refactor this into common utils
    im = Image.fromarray(micro_im.astype(np.uint8))
    im.save(fname)


if __name__ == "__main__":
    # Load user settings
    settings = read_user_settings()

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
    micro_data = read_metadata("metadata.json")

    # Generate Machine Learning data
    generate_ml_dataset(
        (L, h_cell),
        (scale, output_img_size, width_wrt_radius),
        (dataset_dir, input_dir, label_dir),
        micro_data,
    )
