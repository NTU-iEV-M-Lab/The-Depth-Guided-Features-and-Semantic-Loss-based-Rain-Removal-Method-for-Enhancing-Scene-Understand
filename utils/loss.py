"""Loss functions for joint depth estimation and rain/fog restoration.

`TotalLoss` implements the published objective (paper Eq. 9-13):

    total_loss = depth_loss + (l1_loss + perceptual_loss) + semantic_loss

with every term equally weighted. `depth_loss` and `l1_loss` are L1 distances,
`perceptual_loss` compares VGG16 features, and `semantic_loss` is the
cross-entropy between SegFormer predictions on the ground-truth and restored
images. `L2Loss`, `LapLoss`, and `DepthSoomthnessLoss` are alternative losses
kept for reference; they are not used by the default objective.
"""
import numpy as np
import tensorflow as tf
from keras.losses import Reduction
from transformers import TFSegformerForSemanticSegmentation


class TotalLoss:
    """Joint depth + restoration objective: L1 + perceptual + semantic losses."""

    def __init__(self):
        # Frozen SegFormer, used only to measure the semantic loss between images.
        model_checkpoint = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
        self.segformer = TFSegformerForSemanticSegmentation.from_pretrained(
            model_checkpoint, ignore_mismatched_sizes=True
        )
        self.segformer = tf.keras.Sequential([
            tf.keras.Input((256, 512, 3)),
            tf.keras.layers.Permute((3, 1, 2)),
            self.segformer,
            tf.keras.layers.Permute((2, 3, 1)),
            tf.keras.layers.Resizing(256, 512),
            tf.keras.layers.Softmax()
        ])
        self.flat = tf.keras.layers.Flatten()
        self.l1 = L1Loss()

        self.PerceptionLoss = PerceptionLoss()
        self.crossentropy = tf.keras.losses.CategoricalCrossentropy()

    def SegmentationLoss(self, clean_true, clean_pre):
        """Semantic loss: cross-entropy between SegFormer outputs of the two images (Eq. 9)."""
        seg_true = self.segformer(clean_true)
        seg_pre = self.segformer(clean_pre)
        return self.crossentropy(seg_true, seg_pre)

    @tf.function
    def __call__(self, clean_true, clean_pre, depth_true, depth_pre):
        # Restoration loss = pixel L1 + perceptual loss (paper Eq. 13).
        clean_loss = self.l1(clean_true, clean_pre) + self.PerceptionLoss(clean_true, clean_pre)
        # Depth loss = L1 on the predicted depth map (paper Eq. 12).
        depth_loss = self.l1(depth_true, depth_pre)
        # Semantic loss (paper Eq. 9).
        seg_loss = self.SegmentationLoss(clean_true, clean_pre)
        # All three terms are summed with equal weight.
        train_loss = depth_loss + clean_loss + seg_loss
        return train_loss, clean_loss, depth_loss, seg_loss


class PerceptionLoss(tf.losses.Loss):
    """Perceptual loss: squared L2 distance of VGG16 features, averaged over 5 layers."""

    def __init__(self, input_shape=(256, 512, 3), reduction=Reduction.AUTO, name=None):
        super().__init__(reduction, name)
        # VGG16 pretrained on ImageNet (auto-downloaded by Keras).
        self.model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        self.model = tf.keras.Model(
            self.model.input,
            [
                self.model.get_layer('block1_conv2').output,
                self.model.get_layer('block2_conv2').output,
                self.model.get_layer('block3_conv3').output,
                self.model.get_layer('block4_conv3').output,
                self.model.get_layer('block5_conv3').output,
            ]
        )
        self.input_shape = input_shape
        for layer in self.model.layers[:-1]:
            layer.trainable = False

    def content_loss(self, feature_true, feature_pre):
        """Average the per-layer squared feature distance over the five VGG16 layers."""
        loss_arr = []
        for true, pre in zip(feature_true, feature_pre):
            C = true.get_shape().as_list()[-1]
            res = true - pre
            res = tf.square(tf.norm(res, 2, axis=-1))
            loss = tf.reduce_mean(res)
            loss = tf.divide(loss, tf.cast(C, loss.dtype))
            loss_arr.append(loss)
        return tf.reduce_mean(loss_arr)

    def __call__(self, y_true, y_pred, sample_weight=None):
        feature_true = self.model(y_true)
        feature_pre = self.model(y_pred)
        loss = self.content_loss(feature_true, feature_pre)
        return loss


class L1Loss(tf.losses.Loss):
    """Mean absolute error normalized by the channel count (paper Eq. 10)."""

    def __init__(self, reduction=Reduction.AUTO, name=None):
        super().__init__(reduction, name)

    def __call__(self, y_true, y_pred, sample_weight=None):
        C = y_true.get_shape().as_list()[-1]
        res = y_true - y_pred
        res = tf.norm(res, 1, axis=-1)
        loss = tf.reduce_mean(res)
        loss = tf.divide(loss, tf.cast(C, tf.float32))
        return loss


class L2Loss(tf.losses.Loss):
    """Mean squared error normalized by the channel count (alternative loss)."""

    def __init__(self, reduction=Reduction.AUTO, name=None):
        super().__init__(reduction, name)
        self.flat = tf.keras.layers.Flatten()

    def __call__(self, y_true, y_pred):
        C = y_true.get_shape().as_list()[-1]
        res = y_true - y_pred
        res = tf.square(tf.norm(res, 2, axis=-1))
        loss = tf.reduce_mean(res)
        loss = tf.divide(loss, tf.cast(C, loss.dtype))
        return loss


class LapLoss(tf.losses.Loss):
    """Laplacian-pyramid L1 loss (alternative restoration loss, unused by default)."""

    def __init__(self, max_levels=3, reduction=Reduction.AUTO, name=None):
       super().__init__(reduction, name)
       self.max_levels = max_levels

    def gauss_kernel(self, size=5, sigma=1.0):
        grid = np.float32(np.mgrid[0:size,0:size].T)
        gaussian = lambda x: np.exp((x - size//2)**2/(-2*sigma**2))**2
        kernel = np.sum(gaussian(grid), axis=2)
        kernel /= np.sum(kernel)
        return kernel

    def conv_gauss(self, t_input, stride=1, k_size=5, sigma=1.6, repeats=1):
        t_kernel = tf.reshape(tf.constant(self.gauss_kernel(size=k_size, sigma=sigma), tf.float32),
                                [k_size, k_size, 1, 1])
        t_kernel3 = tf.concat([t_kernel]*t_input.get_shape()[3], axis=2)
        t_result = t_input
        for r in range(repeats):
            t_result = tf.nn.depthwise_conv2d(t_result, t_kernel3,
                strides=[1, stride, stride, 1], padding='SAME')
        return t_result

    def make_laplacian_pyramid(self, t_img, max_levels):
        t_pyr = []
        current = t_img
        for level in range(max_levels):
            t_gauss = self.conv_gauss(current, stride=1, k_size=5, sigma=2.0)
            t_diff = current - t_gauss
            t_pyr.append(t_diff)
            current = tf.nn.avg_pool(t_gauss, [1,2,2,1], [1,2,2,1], 'VALID')
        t_pyr.append(current)
        return t_pyr

    def laploss(self, t_img1, t_img2, max_levels=3):
        t_pyr1 = self.make_laplacian_pyramid(t_img1, max_levels)
        t_pyr2 = self.make_laplacian_pyramid(t_img2, max_levels)
        t_losses = [tf.norm(a-b,ord=1)/tf.size(a, out_type=tf.float32) for a,b in zip(t_pyr1, t_pyr2)]
        t_loss = tf.reduce_sum(t_losses)*tf.shape(t_img1, out_type=tf.float32)[0]
        return t_loss

    def call(self, y_true, y_pred):
        return self.laploss(y_true, y_pred, self.max_levels)


class DepthSoomthnessLoss(tf.losses.Loss):
    """Edge-aware depth smoothness loss (alternative depth loss, unused by default)."""

    def __init__(self, reduction=Reduction.AUTO, name=None):
        super().__init__(reduction, name)

    def __call__(self, y_true, y_pred):
       # Edges
        dy_true, dx_true = tf.image.image_gradients(y_true)
        dy_pred, dx_pred = tf.image.image_gradients(y_pred)
        weights_x = tf.exp(tf.reduce_mean(tf.abs(dx_true)))
        weights_y = tf.exp(tf.reduce_mean(tf.abs(dy_true)))

        # Depth smoothness
        smoothness_x = dx_pred * weights_x
        smoothness_y = dy_pred * weights_y

        depth_smoothness_loss = tf.reduce_mean(abs(smoothness_x)) + tf.reduce_mean(
            abs(smoothness_y)
        )

        return depth_smoothness_loss
