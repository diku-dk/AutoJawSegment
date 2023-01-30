import sys

import numpy as np
import os
from scipy.ndimage import distance_transform_edt
import nibabel as nib


def unet_weight_map3D(y, thresh=1):
    no_labels = y == 0
    label_ids = sorted(np.unique(y))[1:]

    distances = np.zeros((y.shape[0], y.shape[1], y.shape[2]))

    for i, label_id in enumerate(label_ids):
        y_cur = np.logical_or(y == label_id, y == 0)
        distances_cur = distance_transform_edt(y_cur).astype(np.float32) <= thresh
        distances_cur = np.logical_and(distances_cur, y == label_id)
        distances = np.logical_or(distances, distances_cur)

    distances = distances.astype(np.uint8)

    return distances


if __name__ == '__main__':

    thresh = 1

    if len(sys.argv) <= 1:
        path1 = '/Users/px/Downloads/AllRes/newGap'
    else:
        path1 = sys.argv[1]

    cat = os.path.basename(path1)
    save = os.path.join(os.path.dirname(path1), f'{cat}_active_weight_map_{thresh}')

    region_sums = []

    if not os.path.exists(save):
        os.mkdir(save)

    for i in os.listdir(path1):
        if i.startswith('.') or not (i.endswith('.nii') or i.endswith('.gz')): continue
        p1 = os.path.join(path1, i)
        p3 = os.path.join(save, i)

        img = nib.load(p1)
        affine = img.affine
        img = img.get_fdata()
        img = img.astype(np.float32)
        res = unet_weight_map3D(img, thresh=thresh)
        region_sums.append(np.sum(res))
        res = nib.Nifti1Image(res, affine)
        nib.save(res, p3)

    print(f'{cat}_mean = {np.mean(region_sums)}')
    print(f'{cat}_std = {np.std(region_sums)}')
