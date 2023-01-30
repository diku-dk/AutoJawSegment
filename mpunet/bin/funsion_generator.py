import os

import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod


class FusionGenerator():
    def __init__(self, batch_size=2**17, base_dir="data_folder/train",
                 n_classes=2
                 ):
        self.n_classes = n_classes
        #super().__init__()

        self.base_dir = base_dir
        self.batch_size = batch_size
        #
        # self.filter_teeth = filter_teeth
        # self.normalize = normalize
        # self.dim = dim
        # # self.resize = resize
        # self.expand_dims_x = expand_dims_x
        # self.expand_dims_y = expand_dims_y
        # self.one_hot = one_hot


    def __getitem__(self, idx):
        images_subdir_list = os.listdir(self.base_dir)
        random_image_subdir = np.random.choice(images_subdir_list, 1)[0]
        image_subdir_path = os.path.join(self.base_dir, random_image_subdir)

        file_names = os.listdir(self.base_dir)

        view_names = list(sorted([f for f in file_names if 'view' in f]))

        label_name = [f for f in file_names if 'label' in f][0]

        view_paths = [os.path.join(image_subdir_path, f) for f in view_names]
        label_path = os.path.join(image_subdir_path, label_name)

        targets = np.loadtxt(label_path,
                           max_rows=self.batch_size,
                           skiprows=None
                           ).astype(np.float32)

        targets = targets.reshape(-1, 1)

        points = np.empty(shape=(len(targets), len(view_paths), self.n_classes),
                          dtype=np.float32)
        points.fill(np.nan)

        for k, v_path in enumerate(view_paths):
            points[:, k, :] = np.loadtxt(v_path,
                           max_rows=self.batch_size,
                           skiprows=None)

        # input = tf.convert_to_tensor(points)
        # output = tf.convert_to_tensor(output)

        return points, targets

    def __len__(self):
        """ Undefined, return some high number - a number is need in keras """
        return int(10**12)

    def __call__(self):
        def tensor_iter():
            """ Iterates the dataset, converting numpy arrays to tensors """
            for x, y in self:
                yield (tf.convert_to_tensor(x),
                       tf.convert_to_tensor(y))
        return tensor_iter()

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

