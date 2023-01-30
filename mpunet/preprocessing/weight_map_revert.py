import sys

import numpy as np
from skimage.io import imshow
# from skimage.measure import label

from scipy.ndimage import label, generate_binary_structure

from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import os

def unet_weight_map3D_revert(y, wc=None, thresh_dist=6):
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

        w = np.ones(d1.shape, dtype=np.float32)

        # gap_regions = np.logical_and(((d1 + d2) < 6), y == 0)
        gap_regions = (d1 + d2) < thresh_dist

        w[gap_regions] = 0
        w -= 1
        # w = ((d1 + d2) > 2).astype(np.float32)
        # w[w == 0] = 0.1
        # w -= 1
        w = w.astype(np.float32)

    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


if __name__ == '__main__':

    images_root = '../../data_folder/jawDecLabelsAll'

    images_root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jawAlllabels'
    label_root = os.path.join(images_root, 'labels')

    # label_root = os.path.join(images_root, 'labels_eroded')
    weight_root = os.path.join(images_root, 'weight_map_revert')

    if not os.path.exists(weight_root):
        os.mkdir(weight_root)

    for i in os.listdir(label_root):
        if i.startswith('.'): continue

        path = os.path.join(label_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img_data = img_func.get_fdata()

        img_data = np.squeeze(img_data)

        img_data = img_data.astype(np.uint8)

        weight_map = unet_weight_map3D_revert(img_data, thresh_dist=3)

        ni_img = nib.Nifti1Image(weight_map, affine)
        save_path = os.path.join(weight_root, i)
        nib.save(ni_img, save_path)
