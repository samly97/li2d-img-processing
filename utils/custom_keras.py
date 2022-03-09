import numpy as np
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
        num_collocation: int,
        name="laplacian",
    ):
        super().__init__(name=name)

        self.num_collocation = num_collocation

        _circ_coords = self._circ_indices(
            width_wrt_radius,
            img_size,
        )
        self._col_points = self._get_collocation_points(
            _circ_coords,
            num_collocation,
        )

        self.list_boxes = tf.map_fn(
            self._get_neighbourhood,
            self._col_points,
        )
        # Trick to get tf.map_fn to work when computing the Laplacian because
        # apparently the thing wants the input type (should be tf.int32 because
        # these are indices) to be of the output type (tf.float32).
        self.list_boxes = tf.cast(self.list_boxes, dtype=tf.float32)

    def call(self, y_true, y_pred):
        _ = y_true

        shape = tf.shape(y_pred)

        batch_size = shape[0]
        batch_size = tf.cast(batch_size, dtype=tf.float32)

        loss = tf.convert_to_tensor(0, dtype=tf.float32)

        for img in y_pred:
            img_fn = self.decorated_compute_laplacian(img)

            img_loss = tf.map_fn(
                img_fn,
                self.list_boxes
            )

            img_loss = tf.cast(img_loss, dtype=tf.float32)
            img_loss = tf.square(img_loss)
            img_loss = tf.reduce_mean(img_loss)

            loss = tf.math.add(loss, img_loss)

        return tf.divide(loss, batch_size)

    def decorated_compute_laplacian(self, img):
        def wrapped(box):
            return self.compute_laplacian(img, box)
        return wrapped

    def compute_laplacian(
        self,
        sample,
        box,
    ):
        laplacian = tf.constant([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ], dtype=tf.float32)
        laplacian = tf.reshape(laplacian, (9,))

        # These are indices so we cast them back to an integer type
        box = tf.cast(box, dtype=tf.int64)

        region = tf.gather_nd(
            sample,
            box
        )
        region = tf.reshape(region, (9,))

        lp = tf.tensordot(
            region,
            laplacian,
            1,
        )

        return lp

    def _get_neighbourhood(self, indices: np.ndarray):
        y = tf.gather(indices, 0)
        x = tf.gather(indices, 1)

        ym1 = tf.subtract(y, 1)
        yp1 = tf.add(y, 1)

        xm1 = tf.subtract(x, 1)
        xp1 = tf.add(x, 1)

        ret = [
            [[ym1, xm1], [ym1, x], [ym1, xp1]],
            [[y, xm1], [y, x], [y, xp1]],
            [[yp1, xm1], [yp1, x], [yp1, xp1]],
        ]
        ret = tf.convert_to_tensor(ret)

        return ret

    def _get_collocation_points(
        self,
        circ_coords: np.ndarray,
        num_collocation: int,
    ) -> np.ndarray:

        random_arr = [i for i in range(circ_coords.shape[0])]
        random_arr = np.array(random_arr)

        np.random.shuffle(random_arr)

        coll_points = circ_coords[random_arr[:num_collocation], :]

        return coll_points

    def _circ_indices(
        self,
        width_wrt_radius: float,
        img_size: int,
    ) -> np.ndarray:
        blank = np.zeros((img_size, img_size), dtype=bool)

        # Center of circle
        c = np.floor(img_size / 2).astype(np.uint32)
        # Radius of circle in pixel units
        r_pix = c / width_wrt_radius

        x_linspace = np.linspace(0, img_size, img_size)
        y_linspace = x_linspace

        xx, yy = np.meshgrid(x_linspace, y_linspace)

        in_circ = (xx - c) ** 2 + (yy - c) ** 2 <= r_pix ** 2
        blank[in_circ] = True

        circ_coords_as_tuple_of_indices = np.where(blank == True)

        circ_coords = np.vstack(
            [
                circ_coords_as_tuple_of_indices[0],
                circ_coords_as_tuple_of_indices[1],
            ]
        )
        circ_coords = circ_coords.T

        # (y, x)
        return circ_coords


class Combined_Loss(tf.keras.losses.Loss):

    def __init__(
        self,
        width_wrt_radius: int,
        inner_width_wrt_radius: float,
        img_size: int,
        num_collocation: int,
        name="combined",
        scale: int = 10,
        regularization: float = 0.05,
    ):
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
            num_collocation,
        )

    def call(self, y_true, y_pred):
        d_loss = self.data_loss.call(y_true, y_pred)
        p_loss = self.physics_loss.call(y_true, y_pred)

        return d_loss + self.reg * p_loss
