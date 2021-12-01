#####################################
## CREATE MACHINE LEARNING DATASET ##
#####################################

## TODO:
## - ACTIVATIONS: seperate geometry from concentration prediction
## - LOCAL porosity
## - incorporate BOUNDING_BOX parameterization
## - zoom ratios

## Current Pseudo Code
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

## Thoughts on Pending Modifications/Additions
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

import os
from typing import Tuple, List

from tqdm import tqdm

import json

from PIL import Image
from skimage import io
from skimage.transform import resize

# Just for testing how images look
import matplotlib.pyplot as plt

if __name__ == "__main__":
	
	# Variable to store image

	# For each microstructure, e.g. micros [1, 2, 3, 4, 5]
	for i in range(1, 6):



















