"""Custom Keras metrics."""
import tensorflow as tf


class Metric(tf.keras.metrics.Metric):
    """Running-mean metric that averages per-batch loss values over an epoch."""

    def __init__(self, name='custom_metric', **kwargs):
        super().__init__(name=name, **kwargs)
        self._loss_list = []

    def update_state(self, loss):
        self._loss_list.append(loss)

    def result(self):
        return tf.reduce_mean(self._loss_list)

    def reset_state(self):
        self._loss_list = []
