"""Plotting helpers: training-curve figures and segmentation-mask colorization."""
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from .callback import LearningRateTracker

# Cityscapes 19-class palette: [class name, label id, RGB color].
COLORMAP = [
#     name            id        color
    ["road",           7,       [128, 64, 128]],
    ["sidewalk",       8,       [244, 35, 232]],
    ["building",      11,       [70, 70, 70]],
    ["wall",          12,       [102, 102, 156]],
    ["fence",         13,       [190, 153, 153]],
    ["pole",          17,       [153, 153, 153]],
    ["traffic light", 19,       [250, 170, 30]],
    ["traffic sign",  20,       [220, 220, 0]],
    ["vegetation",    21,       [107, 142, 35]],
    ["terrain",       22,       [152, 251, 152]],
    ["sky",           23,       [70, 130, 180]],
    ["person",        24,       [220, 20, 60]],
    ["rider",         25,       [255, 0, 0]],
    ["car",           26,       [0, 0, 142]],
    ["truck",         27,       [0, 0, 70]],
    ["bus",           28,       [0, 60, 100]],
    ["train",         31,       [0, 80, 100]],
    ["motorcycle",    32,       [0, 0, 230]],
    ["bicycle",       33,       [119, 11, 32]],
]

def loss_plot(history:tf.keras.callbacks.History, path:str, lr_tracker:LearningRateTracker):
    """Plot loss and metrics.

    Args:
        history: tf.keras.callbacks.History object.
        path: Save plot path.
        lr_tracker: LearningRateTracker holding the per-epoch learning rates.
    """
    name = ['total_loss', 'depth_loss', 'seg_loss', 'restore_loss']
    val_name = ['val_' + var for var in name]

    for i in range(len(name)):
        plt.figure(i)
        plt.plot(history.history[name[i]])
        plt.plot(history.history[val_name[i]])
        plt.title(name[i])
        plt.grid()
        plt.ylabel(name[i])
        plt.xlabel('epochs')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.savefig(os.path.join(path, name[i] + '.png'))
    plt.figure(len(name))
    plt.plot(lr_tracker.lr_arr)
    plt.title('LearningRateTracker')
    plt.grid()
    plt.ylabel('learning rate')
    plt.xlabel('epochs')
    plt.savefig(os.path.join(path, 'learning_rate.png'))
    plt.show()

def predict2rgb(predict):
    """Colorize a class-index segmentation map into an RGB image via COLORMAP."""

    predict = tf.squeeze(predict)


    r = np.zeros_like(predict).astype(np.uint8)
    g = np.zeros_like(predict).astype(np.uint8)
    b = np.zeros_like(predict).astype(np.uint8)

    for i, map in enumerate(COLORMAP):

        # Boolean mask: True where the prediction equals the current class index.
        current_class = (predict == i)

        # RGB triple for the current class.
        r_value, g_value, b_value = map[-1]

        # Paint that class's pixels with its color.
        r[current_class] = r_value
        g[current_class] = g_value
        b[current_class] = b_value

    rgb_mask = np.stack([r, g, b], axis=2)


    return rgb_mask
