import numpy as np
import tensorflow as tf


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

    def inverse(
        self,
        rgb_arr: tf.Tensor,
    ) -> tf.Tensor:
        r''' inverse should work with TensorFlow operations. Take `rgb_arr` as
        a Tensor with [img_size, img_size, 3], reshape to
        [img_size * img_size, 1] for analysis.

        Inputs:
        - rgb_arr: RGB image of [img_size, img_size, 3] as a tf.Tensor with
            `dtype = tf.int32`

        Outputs:
        - sol: State-of-Lithiation vector of [img_size * img_size, 1] as a
            tf.Tensor with `dtype = tf.float32`.

            Note: this does NOT filter out the particle explicity. That has to
                be handled separately.
        '''

        f = tf.cast(self.f, tf.float32)
        f2 = tf.math.multiply(f, f)

        x1 = rgb_arr[..., 0]
        x3 = rgb_arr[..., 2]

        Y, X = x1.shape
        dim = tf.multiply(Y, X)

        x1 = tf.reshape(x1, (dim, 1))
        x3 = tf.reshape(x3, (dim, 1))

        temp = tf.math.multiply(f, x3)
        sol = tf.math.add(x1, temp)
        sol = tf.math.divide(sol, f2)

        return sol
