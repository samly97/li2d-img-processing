import tensorflow as tf
import scipy.ndimage as ndi

from .image import circle_mask_as_vector_of_indices


class Image_Processing_Helpers():

    @staticmethod
    def get_boundaries(mask):
        not_mask = tf.math.logical_not(mask)
        dilated_not_mask = tf.map_fn(
            Image_Processing_Helpers._binary_dilation_tensors,
            not_mask,
        )

        boundary_mask = tf.math.logical_and(dilated_not_mask, mask)
        return boundary_mask

    @staticmethod
    def _binary_dilation_tensors(tensors):
        return tf.py_function(
            lambda tensor: ndi.binary_dilation(tensor),
            [tensors],
            tf.bool,
        )

    @staticmethod
    def get_boundary_stats(boundary_values):
        boundary_mean = tf.map_fn(
            Image_Processing_Helpers._reduce_mean,
            boundary_values,
            fn_output_signature=tf.TensorSpec(
                shape=(), dtype=tf.float32, name=None),
        )
        boundary_std = tf.map_fn(
            Image_Processing_Helpers._reduce_std,
            boundary_values,
            fn_output_signature=tf.TensorSpec(
                shape=(), dtype=tf.float32, name=None),
        )

        return boundary_mean, boundary_std

    @staticmethod
    def get_criteria_tensor(
        boundary_mean,
        boundary_std,
        img_size,
        sigma_factor,
    ):
        criteria = tf.math.subtract(
            boundary_mean,
            tf.math.scalar_mul(sigma_factor, boundary_std),
        )
        criteria = tf.reshape(criteria, (tf.size(criteria), 1, 1, 1))
        criteria = tf.tile(criteria, [1, img_size, img_size, 1])
        return criteria

    #######################################################################
    # HELPERS FUNCTIONS TO `Image_Processing_Helpers.get_boundary_states` #
    #######################################################################

    def _flatten_dims(tensors):
        return tf.py_function(
            lambda tensor: tf.reshape(tensor, [-1]),
            [tensors],
            tf.float32,
        )

    def _reduce_mean(tensors):
        return tf.py_function(
            lambda tensor: tf.math.reduce_mean(
                Image_Processing_Helpers._flatten_dims(tensor),
            ),
            [tensors],
            tf.float32,
        )

    def _reduce_std(tensors):
        return tf.py_function(
            lambda tensor: tf.math.reduce_std(
                Image_Processing_Helpers._flatten_dims(tensor),
            ),
            [tensors],
            tf.float32,
        )


class Mask_MSE(tf.keras.losses.Loss):

    def __init__(
        self,
        img_size: int,
        sigma_factor: float = 1.5,
        name="mask_mask"
    ):
        super().__init__(name=name)
        self.img_size = img_size
        self.sigma_factor = sigma_factor

    def call(self, y_true, y_pred):
        # Evaluate pixels on the intersection where `y_pred` and `y_true` have
        # valid input

        # Just making sure we're not ignoring negatives, since if the
        # predictions are negative within the valid region then we're
        # artificially masking out these regions.
        mask_pred_greater = tf.math.greater(y_pred, 0.0)
        mask_pred_lesser = tf.math.less(y_pred, 0.0)

        mask_pred = tf.math.logical_or(mask_pred_greater, mask_pred_lesser)
        mask_true = tf.math.greater(y_true, 0.0)

        # Intersection between both masks
        mask = tf.math.logical_and(mask_pred, mask_true)

        # Filter for outliers on the boundaries - we presume that there may be
        # artifacts due to the zooming process during data processing of ETL
        # data
        boundary_mask = Image_Processing_Helpers.get_boundaries(mask)
        boundary_mask.set_shape([None, self.img_size, self.img_size, 1])
        boundary_values = tf.ragged.boolean_mask(y_true, boundary_mask)
        (
            boundary_mean,
            boundary_std
        ) = Image_Processing_Helpers.get_boundary_stats(boundary_values)

        # Get criteria to filter boundary values by:
        #   only values > (mean - f * std)
        criteria = Image_Processing_Helpers.get_criteria_tensor(
            boundary_mean,
            boundary_std,
            self.img_size,
            self.sigma_factor,
        )

        # Get boundary values in the same shape as `y_true`
        boundary_mask = tf.cast(boundary_mask, tf.float32)
        boundary_val_img = tf.math.multiply(y_true, boundary_mask)

        # These values are kept along the boundary
        boundary_outlier_mask = tf.math.greater_equal(
            tf.math.subtract(boundary_val_img, criteria),
            0.0,
        )

        # Remove the presumed outliers from the particle mask
        boundary_outlier_mask = tf.cast(boundary_outlier_mask, tf.float32)
        mask = tf.cast(mask, tf.float32)

        mask = tf.math.multiply(mask, boundary_outlier_mask)

        y_pred = tf.math.multiply(y_pred, mask)
        y_true = tf.math.multiply(y_true, mask)

        return tf.math.reduce_mean(tf.square(y_pred - y_true))


class Custom_2D_ROI_MSE(tf.keras.losses.Loss):

    def __init__(
        self,
        width_wrt_radius: float,
        img_size: int,
        scale: int = 10,
        name="custom_mse",
    ):
        super().__init__(name=name)
        self.mask = circle_mask_as_vector_of_indices(
            width_wrt_radius,
            img_size,
            scale,
        )

    def call(self, y_true, y_pred):
        y_true = self._img_to_vector(y_true)
        y_pred = self._img_to_vector(y_pred)

        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        return mse

    def _img_to_vector(self, y):
        shape = tf.shape(y)
        dim = tf.reduce_prod(shape[1:-1])

        y = tf.reshape(y, (-1, dim, shape[-1]))
        y = tf.gather(y, indices=self.mask, axis=1)

        return y
