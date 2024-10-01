from scipy.ndimage import label
import numpy as np

import nibabel as nib
import os

from connected_component_3D import connected_component_3D
from morphologies import binary_closing_label

def get_max_label(labels):
    values, counts = np.unique(labels, return_counts=True)
    inds = np.argsort(counts)

    for i in reversed(inds):
        if values[i] == 0:
            continue
        else:
            res = values[i]
            return res


def symmetric_separator(file_path, save_path=None, connectivity=26, portion_foreground=0.01,
                                    bg_val=0, pairs=None, no_save=False, morph=False):

    img_func = nib.load(file_path)
    affine = img_func.affine
    img = img_func.get_fdata().astype(np.uint8)
    img = np.squeeze(img)
    # res = np.empty(img.shape)

    for (i, j) in pairs:
        img_i = np.copy(img == i)
        img_j = np.copy(img == j)

        img[img_i] = 0
        img[img_j] = 0

        labels_out_i, _ = label(img_i)
        max_label = get_max_label(labels_out_i)

        print(f'max_label of class {i} is {max_label}')

        img[np.logical_and(labels_out_i > 0, labels_out_i != max_label)] = j
        img[labels_out_i == max_label] = i

        labels_out_j, _ = label(img_j)
        max_label = get_max_label(labels_out_j)

        print(f'max_label of class {j} is {max_label}')

        img[np.logical_and(labels_out_j > 0, labels_out_j != max_label)] = i
        img[labels_out_j == max_label] = j

    img = connected_component_3D(img, connectivity=26, portion_foreground=portion_foreground)

    if morph:
        img = binary_closing_label(img, radius=3)

    if no_save:
        return img

    ni_img = nib.Nifti1Image(img.astype(np.uint8)
                             , affine)

    nib.save(ni_img, save_path)


if __name__ == '__main__':

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_weight_map/predictions_0902/nii_files'

    ccd_path = os.path.join(root_path, 'ccd_symmetric')

    if not os.path.exists(ccd_path):
        os.mkdir(ccd_path)

    for i in reversed(sorted(os.listdir(root_path))):
        print(f'working on {i}')
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue

        img_path = os.path.join(root_path, i)

        save_path = os.path.join(ccd_path, i)

        portion_foreground = 0.3

        symmetric_separator(img_path, save_path=save_path,
                            connectivity=6, portion_foreground=portion_foreground,
                            pairs=[(1, 2), (3, 4)],
                            morph=True
                            )

