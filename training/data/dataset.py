import tensorflow as tf 
import numpy as np
import math
import os

from tensorflow.keras.utils import Sequence
from skimage.io import imread
from skimage.transform import resize

class SegmentationDataset(Sequence):
    def __init__(self, image_dir, label_dir, batch_size, augmentation=False, resize=None, **kwargs):
        super().__init__(**kwargs)
        self.images, self.gt_segmentations = _get_paths(image_dir, label_dir)

        self.batch_size = len(self.images) if batch_size == -1 else batch_size
        self.augmentation = augmentation
        self.resize = resize

    def __len__(self):
        return int(math.ceil(len(self.images) / self.batch_size))

    def get_batch_image(self, idx):
        batch_images = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([resize(imread(f), self.resize) for f in batch_images]) if self.resize else np.array([imread(f) for f in batch_images])

    def get_batch_gt_segmentation(self, idx):
        batch_masks = self.gt_segmentations[idx * self.batch_size:(idx + 1) * self.batch_size]
        return np.array([resize(imread(f, as_gray=True), self.resize) for f in batch_masks]) if self.resize else np.array([imread(f, as_gray=True) for f in batch_masks])

    def __getitem__(self, index):
        return self.get_batch_image(index), self.get_batch_gt_segmentation(index)

def _get_paths(images_directory, labels_directory):
    image_files = sorted([f for f in os.listdir(images_directory) if f.endswith(".jpg") and "_mask" not in f])
    images_path, labels_path = [], []

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        mask_file = base_name + "_mask.png"
        img_path = os.path.join(images_directory, img_file)
        mask_path = os.path.join(labels_directory, mask_file)

        if os.path.exists(mask_path):
            images_path.append(img_path)
            labels_path.append(mask_path)
        else:
            print(f"⚠️ Skipping: No mask found for {img_file}")

    return images_path, labels_path
