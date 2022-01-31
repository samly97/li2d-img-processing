import tensorflow as tf

from .image import circle_mask_as_vector_of_indices


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
        shape = y.shape
        dim = tf.reduce_prod(shape[1:-1])

        y = tf.reshape(y, (-1, dim, shape[-1]))
        y = tf.gather(y, indices=self.mask, axis=1)

        return y
