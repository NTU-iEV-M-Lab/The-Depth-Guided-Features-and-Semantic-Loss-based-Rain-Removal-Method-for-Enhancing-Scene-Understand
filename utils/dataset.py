"""RainCityscapes dataset loader.

Resolves the matched rain / clear / depth / label file lists for the
RainCityscapes benchmark and eagerly decodes them into NumPy arrays for
training and evaluation.
"""
import os
import time
import warnings
from glob import glob

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import TqdmExperimentalWarning
from tqdm.rich import tqdm

from .chronograph import timer_func

warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)

class RainCityscapes:
    """Loads matched rain/clear/depth/label image sets from the RainCityscapes layout."""

    def __init__(
            self,
            image_size=(256, 512),
            rain_dir='data/rain_cityscapes/leftImg8bit_rain/',
            clear_dir='data/cityscapes/leftImg8bit/',
            label_dir='data/cityscapes/gtFine/',
            depth_dir='data/disparity/',
            downsample=3,
            crop_and_resize=True,
            seed=1
    ):
        self.image_size = image_size
        self.rain_dir = rain_dir
        self.clear_dir = clear_dir
        self.label_dir = label_dir
        self.depth_dir = depth_dir
        self._helper_mode = 'img'
        self.seed = seed
        self.downsample = downsample
        self.crop_and_resize = crop_and_resize
        self._get_filename()

    @timer_func
    def _get_filename(self):
        """Build the matched train/val file lists for rain, clear, depth, and labels."""
        self.train_rain = list(sorted(glob(self.rain_dir + 'train/*/*.png')))[::self.downsample]
        split = [var.split('/')[-1] for var in self.train_rain]
        name = [var.split('_rain_')[0] for var in split]
        self.train_label = [os.path.join(self.label_dir, 'train', var.split('_')[0], var.split('_left')[0] + '_gtFine_labelTrainIds.png') for var in name]
        self.train_clear = [os.path.join(self.clear_dir, 'train', var.split('_')[0], var + '.png') for var in name]
        self.train_depth = [os.path.join(self.depth_dir, 'train', var.split('_')[0], var.split('_left')[0] + '_disparity.png') for var in name]

        self.train_rain = np.array(self.train_rain)
        self.train_label = np.array(self.train_label)
        self.train_clear = np.array(self.train_clear)
        self.train_depth = np.array(self.train_depth)

        self.test_rain = list(sorted(glob(self.rain_dir + 'val/*/*.png')))[::self.downsample]
        split = [var.split('/')[-1] for var in self.test_rain]
        name = [var.split('_rain_')[0] for var in split]
        self.test_label = [os.path.join(self.label_dir, 'val', var.split('_')[0], var.split('_left')[0] + '_gtFine_labelTrainIds.png') for var in name]
        self.test_clear = [os.path.join(self.clear_dir, 'val', var.split('_')[0], var + '.png') for var in name]
        self.test_depth = [os.path.join(self.depth_dir, 'val', var.split('_')[0], var.split('_left')[0] + '_disparity.png') for var in name]

    def read_file(self, path):
        """Decode and resize one file; the mode is selected by `self._helper_mode`."""
        image = tf.io.read_file(path)
        if self._helper_mode == 'img':
            image = tf.image.decode_png(image, channels=3)
            # image = tf.cast(image, tf.float32) / 255.
            image = (tf.cast(image, tf.float32) / 127.5) - 1.0
            image = tf.expand_dims(image, 0)
            if self.crop_and_resize:
                image = tf.image.crop_and_resize(image, [[0.04, 0.05, 0.82, 0.99]],box_indices=[0], crop_size=[self.image_size[0], self.image_size[1]])[0]
            else:
                image = tf.image.resize(image, (self.image_size[0], self.image_size[1]))[0]
            
        elif self._helper_mode == 'depth':
            image = tf.image.decode_png(image, channels=1)
            image = tf.expand_dims(image, 0)
            if self.crop_and_resize:
                image = tf.image.crop_and_resize(image, [[0.04, 0.05, 0.82, 0.99]],box_indices=[0], crop_size=[self.image_size[0], self.image_size[1]])[0]
            else:
                image = tf.image.resize(image, (self.image_size[0], self.image_size[1]))[0]
            image = tf.cast(image, tf.float32) / 255.

        elif self._helper_mode == 'label':
            image = tf.image.decode_png(image, channels=1)
            image = tf.expand_dims(image, 0)
            if self.crop_and_resize:
                image = tf.image.crop_and_resize(image, [[0.04, 0.05, 0.82, 0.99]],box_indices=[0], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
                                              crop_size=[self.image_size[0], self.image_size[1]])[0]
            else:
                image = tf.image.resize(image, (self.image_size[0], self.image_size[1]))[0]
        else:
            print("File read mode error.")
            exit()
        return image
    
    def file_list_to_dataset(self, pack):
        """Decode a (rain, clear, depth, label) tuple of file lists into NumPy arrays."""
        rain, clear , depth, label = pack
 
        rain = tf.data.Dataset.from_tensor_slices(rain)
        clear = tf.data.Dataset.from_tensor_slices(clear)

        depth = tf.data.Dataset.from_tensor_slices(depth)
        label = tf.data.Dataset.from_tensor_slices(label)
        self._helper_mode = 'img'
        rain = rain.map(self.read_file, num_parallel_calls=tf.data.AUTOTUNE)
        rain = np.array(list(tfds.as_numpy(rain)))
        self.bar.update(self.update_step)
        time.sleep(0.01)

        clear = clear.map(self.read_file, num_parallel_calls=tf.data.AUTOTUNE)
        clear = np.array(list(tfds.as_numpy(clear)))
        self.bar.update(self.update_step)
        time.sleep(0.01)

        self._helper_mode = 'depth'
        depth = depth.map(self.read_file, num_parallel_calls=tf.data.AUTOTUNE)
        depth = np.array(list(tfds.as_numpy(depth)))
        self.bar.update(self.update_step)
        time.sleep(0.01)

        self._helper_mode = 'label'
        label = label.map(self.read_file, num_parallel_calls=tf.data.AUTOTUNE)
        label = np.array(list(tfds.as_numpy(label)))
        self.bar.update(self.update_step)
        time.sleep(0.01)

        return rain, clear, depth, label
    
    @timer_func
    def gernerate_dataset(self):
        """Load and shuffle the training split as decoded NumPy arrays."""
        print('Loading Dataset for training and validation...')
        self.bar = tqdm(total=4)
        
        index = list(range(len(self.train_rain)))
        np.random.seed(self.seed)
        np.random.shuffle(index)
        self.update_step = 1

        train = [self.train_rain[index], self.train_clear[index], self.train_depth[index], self.train_label[index]]
        train_rain, train_clear, train_depth, train_label = self.file_list_to_dataset(train)
        self.bar.close()
        return train_rain, train_clear, train_depth, train_label

    @timer_func
    def generate_testset(self):
        """Load the validation/test split as decoded NumPy arrays."""
        print('Loading Dataset for testing...')
        self.bar = tqdm(total=4)
        
        self.update_step = 1

        test = [self.test_rain, self.test_clear, self.test_depth, self.test_label,]
        test_rain, test_clear, test_depth, test_label = self.file_list_to_dataset(test)
        self.bar.close()
        return test_rain, test_clear, test_depth, test_label
    