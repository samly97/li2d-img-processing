import tensorflow as tf

from .image import circle_mask_as_vector_of_indices


class Mask_MSE(tf.keras.losses.Loss):

    def __init__(self, name="mask_mask"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        # Evaluate pixels on the intersection where `y_pred` and `y_true` have
        # valid input
        mask_pred = tf.math.greater(y_pred, 0.0)
        mask_true = tf.math.greater(y_true, 0.0)

        # Intersection between both masks
        mask = tf.math.logical_and(mask_pred, mask_true)
        mask = tf.cast(mask, tf.float32)

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
