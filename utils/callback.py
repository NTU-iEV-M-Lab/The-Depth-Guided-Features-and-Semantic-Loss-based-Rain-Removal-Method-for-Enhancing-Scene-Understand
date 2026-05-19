"""Training callbacks."""
import tensorflow as tf


class LearningRateTracker(tf.keras.callbacks.Callback):
    """Records the learning rate at the start of every epoch for later plotting."""

    def __init__(self):
        super().__init__()
        self.lr_arr = []

    def on_epoch_begin(self, epoch, logs=None):
        optimizer = self.model.optimizer
        _lr = tf.keras.backend.get_value(optimizer.lr)
        self.lr_arr.append(_lr)
        print(f'\nEpoch {epoch+1}: Learning rate: {_lr}.')
