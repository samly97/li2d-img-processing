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
        shape = tf.shape(y)
        dim = tf.reduce_prod(shape[1:-1])

        y = tf.reshape(y, (-1, dim, shape[-1]))
        y = tf.gather(y, indices=self.mask, axis=1)

        return y


class Laplacian_MSE(tf.keras.losses.Loss):

    def __init__(
        self,
        width_wrt_radius: float,
        img_size: int,
        scale: int = 10,
        name="laplacian",
    ):
        super().__init__(name=name)

        self.mask = circle_mask_as_vector_of_indices(
            width_wrt_radius,
            img_size,
            scale,
        )

        # Declare the Laplacian stencil
        self.laplacian = tf.constant([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]], tf.float32)
        self.laplacian = tf.expand_dims(self.laplacian, axis=-1)
        self.laplacian = tf.expand_dims(self.laplacian, axis=-1)

    def call(self, y_true, y_pred):

        _ = y_true

        # Compute laplacian on predicted image
        y_pred = tf.nn.conv2d(y_pred, self.laplacian, 1, padding="VALID")

        # Extract relevant pixels (in circle)
        y_pred = self._img_to_vector(y_pred)

        return tf.reduce_mean(tf.square(y_pred))

    def _img_to_vector(self, y):
        shape = tf.shape(y)
        dim = tf.reduce_prod(shape[1:-1])

        y = tf.reshape(y, (-1, dim, shape[-1]))
        y = tf.gather(y, indices=self.mask, axis=1)

        return y


class Combined_Loss(tf.keras.losses.Loss):

    def __init__(
        self,
        width_wrt_radius: int,
        inner_width_wrt_radius: float,
        img_size: int,
        scale: int = 10,
        regularization: float = 0.05,
        name="combined",
    ):
        # inner_width_wrt_radius can be different than the "original" one
        # as the output image may be formatted differently from input images

        super().__init__(name=name)

        self.reg = regularization

        self.data_loss = Custom_2D_ROI_MSE(
            width_wrt_radius,
            img_size,
            scale,
        )
        self.physics_loss = Laplacian_MSE(
            inner_width_wrt_radius,
            img_size,
            scale,
        )

    def call(self, y_true, y_pred):
        d_loss = self.data_loss.call(y_true, y_pred)
        p_loss = self.physics_loss.call(y_true, y_pred)

        return d_loss + self.reg * p_loss
