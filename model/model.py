"""End-to-end rain/fog restoration model.

`Build_model` wires the PoolFormer encoder to the depth and de-weather decoder
heads. `CustomModel` wraps it with the training/evaluation loop: paired data
augmentation, the joint depth + restoration loss, and metric tracking.
"""
import tensorflow as tf

from model.depth_head import DepthHead_v7
from model.poolformer import Poolformer
from model.weather_head import WeatherHead_v10
from utils import loss, metrics


def Build_model(input_shape=(128, 256, 3)):
    """Build the functional model: encoder -> depth head + depth-guided de-weather head."""
    input = tf.keras.Input(input_shape)
    encode = Poolformer(name='PoolViT_output')(input)
    depth = DepthHead_v7(name='Depth_output')(encode)
    image = WeatherHead_v10(name='Deweather_output')(encode, depth, input)
    model = tf.keras.Model(inputs=input, outputs=[depth, image])

    return model


class CustomModel(tf.keras.Model):
    """Training wrapper: runs augmentation, the joint loss, and metric tracking."""

    def __init__(self, input_shape=(256, 512, 3), seed=1, image_shape=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weather = Build_model(input_shape)

        self.image_shape = input_shape

        # Paired horizontal flip: the shared seed keeps rain/clear/depth aligned.
        self.flip_rain = tf.keras.layers.RandomFlip('horizontal', seed=seed)
        self.flip_clear = tf.keras.layers.RandomFlip('horizontal', seed=seed)
        self.flip_depth = tf.keras.layers.RandomFlip('horizontal', seed=seed)

        self.loss_function = loss.TotalLoss()
        self.depth_metric = metrics.Metric(name='depth_metric')
        self.restore_metric = metrics.Metric(name='restore_metric')
        self.seg_metric = metrics.Metric(name='seg_metric')

    def train_step(self, data):
        """One optimization step on the joint depth + restoration loss."""
        self.depth_metric.reset_state()
        self.restore_metric.reset_state()
        self.seg_metric.reset_state()
        rain, [depth_true, clean_true] = data

        rain = self.flip_rain(rain)
        depth_true = self.flip_depth(depth_true)
        clean_true = self.flip_clear(clean_true)

        with tf.GradientTape() as tape:
            depth_pre, clean_pre = self.weather(rain)
            train_loss, clean_loss, depth_loss, seg_loss = self.loss_function(
                clean_true, clean_pre, depth_true, depth_pre
            )

        # Compute gradients and update weights.
        gradients = tape.gradient(train_loss, self.weather.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.weather.trainable_weights))
        return {
            'total_loss': train_loss, 'depth_loss': depth_loss,
            'restore_loss': clean_loss, 'seg_loss': seg_loss,
        }

    def test_step(self, data):
        """Validation step: evaluate the joint loss without updating weights."""
        rain, [depth_true, clean_true] = data
        depth_pre, clean_pre = self.weather(rain)

        _, restore_loss, depth_loss, seg_loss = self.loss_function(
            clean_true, clean_pre, depth_true, depth_pre
        )
        self.depth_metric.update_state(depth_loss)
        self.restore_metric.update_state(restore_loss)
        self.seg_metric.update_state(seg_loss)

        depth_loss = self.depth_metric.result()
        clean_loss = self.restore_metric.result()
        seg_loss = self.seg_metric.result()
        val_loss = depth_loss + clean_loss + seg_loss

        return {
            'total_loss': val_loss, 'depth_loss': depth_loss,
            'restore_loss': clean_loss, 'seg_loss': seg_loss,
        }

    def __call__(self, rain, **kwargs):
        """Run inference, returning the predicted depth map and restored image."""
        depth_pre, clean_pre = self.weather(rain)
        return depth_pre, clean_pre
