"""Evaluate the restoration model on RainCityscapes.

Computes PSNR / SSIM / depth-L1 over the validation split and, optionally,
saves the predicted (derained) images to disk.

Example:
    python evaluate.py --checkpoint checkpoints/seed2/model.keras
    python evaluate.py --checkpoint checkpoints/seed2/model.keras \\
        --save-images --output-dir results
"""
import argparse
import os

import numpy as np
import tensorflow as tf
from tqdm.rich import tqdm

# Imported so the custom model/layer classes are registered for deserialization.
from model.model import CustomModel  # noqa: F401
from utils import dataset

# Model input resolution (height, width).
IMAGE_SIZE = (256, 512)
# Per-GPU memory cap (MB) requested from TensorFlow; adjust to your hardware.
GPU_MEMORY_LIMIT = 5000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate the restoration model on RainCityscapes."
    )
    parser.add_argument('--checkpoint', required=True,
                        help="Path to the trained model checkpoint (the CustomModel .keras file).")
    parser.add_argument('--save-images', action='store_true',
                        help="Also save the predicted (derained) images to --output-dir.")
    parser.add_argument('--output-dir', default='results',
                        help="Directory for saved prediction images (used with --save-images).")
    return parser.parse_args()


def configure_gpu():
    """Cap GPU memory so the process leaves room for other jobs on the device."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=GPU_MEMORY_LIMIT)])
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized.
            print(e)


def evaluate(model, rain, clean, depth):
    """Compute PSNR / SSIM / depth-L1 statistics over the test split."""
    ssim = np.array([])
    psnr = np.array([])
    depthloss = np.array([])
    l1 = tf.keras.losses.MeanAbsoluteError()

    print('Evaluating SSIM and PSNR on the test dataset.')
    for rain_img, clear_img, depth_img in tqdm(zip(rain, clean, depth), total=len(rain)):
        depth_pre, clear_pre = model.predict(rain_img, verbose=0)

        # Images are in [-1, 1]; rescale to [0, 255] before computing PSNR/SSIM.
        clear_img = tf.cast(clear_img + 1., tf.float32) * 127.5
        clear_pre = tf.cast(clear_pre + 1., tf.float32) * 127.5

        psnr = np.append(psnr, tf.image.psnr(clear_img, clear_pre, 255).numpy())
        ssim = np.append(ssim, tf.image.ssim(clear_img, clear_pre, 255).numpy())
        depthloss = np.append(depthloss, l1(depth_img, depth_pre).numpy())

    print(f'depth_loss : {np.mean(depthloss)}')
    print(f'mean SSIM  : {np.mean(ssim)}')
    print(f'mean PSNR  : {np.mean(psnr)}')
    print(f'median SSIM: {np.median(ssim)}')
    print(f'median PSNR: {np.median(psnr)}')


def save_predictions(model, rain, output_dir):
    """Run the model and save the predicted derained images to `output_dir`."""
    os.makedirs(output_dir, exist_ok=True)
    print(f'Saving predicted images to {output_dir}.')
    count = 0
    for rain_batch in tqdm(rain):
        _, clear_pre = model.predict(rain_batch, verbose=0)
        # Convert from [-1, 1] back to the [0, 1] range expected by save_img.
        clear_pre = (tf.cast(clear_pre + 1., tf.float32) / 2.).numpy()
        for image in clear_pre:
            tf.keras.utils.save_img(os.path.join(output_dir, f'pred_{count:04d}.png'), image)
            count += 1


if __name__ == '__main__':
    args = parse_args()
    tf.config.run_functions_eagerly(True)
    configure_gpu()

    # Load the trained checkpoint and keep only the inference (`weather`) model.
    model = tf.keras.saving.load_model(args.checkpoint, compile=False)
    model = model.weather
    model.summary()

    # Load the RainCityscapes test split and group it into batches of 4.
    raincityscapes = dataset.RainCityscapes(IMAGE_SIZE, downsample=1)
    test_rain, test_clear, test_depth, test_label = raincityscapes.generate_testset()
    with tf.device("CPU:0"):
        rain_split = tf.split(test_rain, len(test_rain) // 4)
        clear_split = tf.split(test_clear, len(test_clear) // 4)
        depth_split = tf.split(test_depth, len(test_depth) // 4)

    evaluate(model, rain_split, clear_split, depth_split)

    if args.save_images:
        save_predictions(model, rain_split, args.output_dir)
