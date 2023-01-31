import sys

import numpy as np
from scipy.ndimage import distance_transform_edt
import nibabel as nib
import os

def unet_weight_map2D(y, wc=None, w0=10, sigma=5):
    # labels = label(y)
    labels = y
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)), dtype=np.float32)

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=-1)
        d1 = distances[..., 0]
        d2 = distances[..., 1]
        w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


def unet_weight_map3D(y, wc=None, w0=10, sigma=5):
    no_labels = y == 0
    label_ids = sorted(np.unique(y))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], y.shape[2], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[..., i] = distance_transform_edt(y != label_id).astype(np.float32)

        distances = distances.astype(np.float32)

        distances = np.sort(distances, axis=-1)
        d1 = distances[..., 0]
        d2 = distances[..., 1]

        del distances

        w = w0 * np.exp(-1 / 2 * ((d1 + d2) ** 2 + (d1 - d2) ** 2 * 0
                                  ) /
                        sigma ** 2) * no_labels

        w = w.astype(np.float32)

        w[w < 0.05] = 0

    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


if __name__ == '__main__':

    images_root = '/Users/px/Downloads/OneDrive_1_16-11-2022'

    label_root = os.path.join(images_root, 'labels')
    weight_root = os.path.join(images_root, 'weight_maps')

    morph = False

    if not os.path.exists(weight_root):
        os.mkdir(weight_root)

    for i in os.listdir(label_root):
        if i.startswith('.'): continue

        path = os.path.join(label_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img_data = img_func.get_fdata()

        img_data = np.squeeze(img_data)

        img_data = img_data.astype(np.float32)

        weight_map = unet_weight_map3D(img_data, w0=10, sigma=3)
        weight_map = weight_map.astype(np.float32)
        ni_img = nib.Nifti1Image(weight_map, affine)
        save_path = os.path.join(weight_root, i)
        nib.save(ni_img, save_path)
