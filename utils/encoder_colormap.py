import numpy as np


class colormap():
    r''' This colormap class is used to encode State-of-Lithiation into RGB
    values since RGB files could be stored efficiently (in terms of memory).

    The encoding process will use the available values in RGB channels, namely
    [0, 255] using the modulus and remainders:

    - with remainder: [r, 0, (x-r) / f]
    - without remainder: [0, 0, x / f]

    Where:
    - f = 255
    - r = x % f (when x % f != 0)

    This will result in a State-of-Lithiation resolution of:
    - 1/65025 * 100 %

    The inverse operation is simply:
    - x1 + f * x3 = SOL
    '''

    def __init__(self, f=255):
        self.f = f

    def __call__(self, sol: np.array):
        r''' Map State-of-Lithiation to RGB values
        '''

        f = self.f
        f2 = f ** 2

        # Reshape into vector
        num_values = sol.size
        sol = sol.reshape(num_values)

        sol = sol * f2
        scaled_sol = np.floor(sol)

        # Get the indices where there are and are not remainders
        modded = (scaled_sol % f).astype(np.uint8)
        remainders = modded != [0]
        no_remainders = ~remainders

        # Set values
        x1_r = scaled_sol[remainders] % f
        x3_r = (scaled_sol[remainders] - x1_r) / f

        # Instantiate ret array
        rgb_im = np.zeros((num_values, 3), dtype=np.uint8)

        rgb_im[remainders, 0] = x1_r
        rgb_im[remainders, 2] = x3_r
        rgb_im[no_remainders, 2] = scaled_sol[no_remainders] / f

        return rgb_im

    def inverse(self, rgb_arr: np.array):
        r''' Get the State-of-Lithiation from RGB values.
        '''

        f = self.f
        f2 = f ** 2

        sol = (rgb_arr[:, :, 0].astype(np.uint32) +
               f * rgb_arr[:, :, 2].astype(np.uint32))
        sol = (sol / f2).astype(np.float32)

        return sol
