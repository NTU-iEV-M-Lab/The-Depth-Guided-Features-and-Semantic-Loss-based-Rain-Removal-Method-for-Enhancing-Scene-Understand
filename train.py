"""Training entry point.

Trains the joint depth + rain/fog restoration model on the RainCityscapes
dataset, saves the best checkpoint, plots the loss curves, and exports the
inner `weather` model for inference.

Example:
    python train.py --epochs 150 --batch-size 4 --lr 3e-4 --seed 2
"""
import argparse
import os

import tensorflow as tf
from model.model import CustomModel
from utils import callback, dataset, plot

# Model input resolution (height, width); fixed by the architecture.
IMAGE_SIZE = (256, 512)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the rain/fog restoration model on RainCityscapes."
    )
    parser.add_argument('--epochs', type=int, default=150,
                        help="Number of training epochs.")
    parser.add_argument('--batch-size', type=int, default=4,
                        help="Mini-batch size.")
    parser.add_argument('--lr', type=float, default=3e-4,
                        help="Peak (base) learning rate for the cosine schedule.")
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help="Weight decay for the Lion optimizer.")
    parser.add_argument('--warmup', type=float, default=0.0,
                        help="Warm-up fraction of total training steps (0 disables warm-up).")
    parser.add_argument('--seed', type=int, default=2,
                        help="Random seed.")
    parser.add_argument('--checkpoint-dir', default='checkpoints',
                        help="Root directory for saved checkpoints.")
    parser.add_argument('--gpu-memory-limit', type=int, default=20480,
                        help="Per-GPU memory cap in MB requested from TensorFlow.")
    return parser.parse_args()


def main(args):
    tf.random.set_seed(args.seed)

    # Cap GPU memory so the process leaves room for other jobs on the device.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=args.gpu_memory_limit)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized.
            print(e)
    tf.keras.backend.clear_session()

    checkpoint_dir = os.path.join(args.checkpoint_dir, f'seed{args.seed}')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load the RainCityscapes train/val splits.
    rain_cityscapes = dataset.RainCityscapes(image_size=IMAGE_SIZE, downsample=3, seed=args.seed)
    train_rain, train_clear, train_depth, train_label = rain_cityscapes.gernerate_dataset()
    val_rain, val_clear, val_depth, val_label = rain_cityscapes.generate_testset()

    # Cosine-decay learning-rate schedule with optional warm-up.
    total_steps = int(len(train_rain) / args.batch_size * args.epochs)
    warmup_steps = int(total_steps * args.warmup)
    scheduled_lrs = tf.keras.optimizers.schedules.CosineDecay(
        1e-6, total_steps, warmup_target=args.lr, warmup_steps=warmup_steps)
    optimizer = tf.keras.optimizers.Lion(scheduled_lrs, weight_decay=args.weight_decay)

    tracker = callback.LearningRateTracker()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=10, monitor='val_total_loss'),
        tracker,
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(checkpoint_dir, 'model.keras'),
            monitor='val_total_loss', save_best_only=True, verbose=1, save_format="keras"),
    ]

    train_protocol = CustomModel((*IMAGE_SIZE, 3), seed=args.seed)
    train_protocol.weather.summary()

    # Keep the large image tensors in CPU memory; batches are moved to GPU during fit.
    with tf.device('/CPU:0'):
        train_rain = tf.constant(train_rain)
        train_depth = tf.constant(train_depth)
        train_clear = tf.constant(train_clear)
        val_rain = tf.constant(val_rain)
        val_depth = tf.constant(val_depth)
        val_clear = tf.constant(val_clear)

    train_protocol.compile(optimizer=optimizer)
    history = train_protocol.fit(
        x=train_rain, y=[train_depth, train_clear], batch_size=args.batch_size, shuffle=True,
        epochs=args.epochs, callbacks=callbacks,
        validation_data=(val_rain, [val_depth, val_clear]))
    plot.loss_plot(history, checkpoint_dir, lr_tracker=tracker)

    # Reload the best checkpoint and export the inner `weather` model for inference.
    model = tf.keras.saving.load_model(os.path.join(checkpoint_dir, 'model.keras'), compile=False)
    model = model.weather
    tf.keras.saving.save_model(model, os.path.join(checkpoint_dir, 'weather.keras'))


if __name__ == '__main__':
    main(parse_args())
