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

if __name__ == "__main__":
	pass