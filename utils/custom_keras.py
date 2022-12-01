import tensorflow as tf


class Mask_MSE(tf.keras.losses.Loss):

    def __init__(self, name="mask_mask"):
        super().__init__(name=name)

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
        mask = tf.cast(mask, tf.float32)

        y_pred = tf.math.multiply(y_pred, mask)
        y_true = tf.math.multiply(y_true, mask)

        return tf.math.reduce_mean(tf.square(y_pred - y_true))
