from scipy.ndimage import label, generate_binary_structure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import os
# import cc3d

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

        # img += labels_out_i.astype(np.uint8)
        # img += labels_out_j.astype(np.uint8)

    img = connected_component_3D(img, connectivity=26, portion_foreground=portion_foreground)

    if morph:
        img = binary_closing_label(img, radius=3)

    if no_save:
        return img

    ni_img = nib.Nifti1Image(img.astype(np.uint8)
                             , affine)

    nib.save(ni_img, save_path)



def symmetric_separator2(file_path, save_path=None, connectivity=26, portion_foreground=0.01,
                                    bg_val=0, pairs=None, no_save=False, morph=False):

    img_func = nib.load(file_path)
    affine = img_func.affine
    img = img_func.get_fdata().astype(np.uint8)
    img = np.squeeze(img)
    res = np.zeros(img.shape)

    for (i, j) in pairs:
        img_i = np.copy(img == i)
        img_j = np.copy(img == j)

        labels_out_i, _ = label(img_i)
        num_total_voxels = np.sum(labels_out_i != 0)

        for m in np.unique(labels_out_i):
            if m == 0:
                continue
            num_voxels = np.sum(labels_out_i == m)
            if m < 10 or m % 10 == 0:
                print(f'num for label {i} part {m} = {num_voxels}')

            if num_voxels > num_total_voxels * portion_foreground:
                res += (i * (labels_out_i == m)).astype(np.uint8)
            else:
                res += (j * (labels_out_i == m)).astype(np.uint8)

        labels_out_j, _ = label(img_j)
        max_label = get_max_label(labels_out_j)

        for n in np.unique(labels_out_j):
            if n == 0:
                continue
            num_voxels = np.sum(labels_out_j == n)
            if n < 10 or n % 10 == 0:
                print(f'num for label {j} part {n} = {num_voxels}')

            if num_voxels > num_total_voxels * portion_foreground:
                res += (j * (labels_out_j == n)).astype(np.uint8)
            else:
                res += (i * (labels_out_j == n)).astype(np.uint8)

    res = connected_component_3D(res, connectivity=26, portion_foreground=portion_foreground)

    if no_save:
        return res

    if morph:
        res = binary_closing_label(res, radius=3)


    ni_img = nib.Nifti1Image(res.astype(np.uint8)
                             , affine)

    nib.save(ni_img, save_path)



if __name__ == '__main__':

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_weight_map/predictions_0902/nii_files'

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/pred_traing_w_2/no_weight_2_2'
    root_path = '/Users/px/Downloads/pred'

    ccd_path = os.path.join(root_path, 'ccd_symmetric')

    if not os.path.exists(ccd_path):
        os.mkdir(ccd_path)

    for i in reversed(sorted(os.listdir(root_path))):
        print(f'working on {i}')
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue

        img_path = os.path.join(root_path, i)

        save_path = os.path.join(ccd_path, i)

        # if os.path.exists(save_path):
        #     print(f'ignoring {i}')
        #     continue

        portion_foreground = 0.01

        portion_foreground = 0.3

        # portion_foreground = {1: 0.3, 2: 0.01}

        #
        # symmetric_separator2(img_path, save_path=save_path,
        #                     connectivity=6, portion_foreground=portion_foreground,
        #                     pairs=[(3, 2), (1, 4)],
        #                     morph=False,
        #                     # pairs=[(4, 2)]
        #                     )


        symmetric_separator(img_path, save_path=save_path,
                            connectivity=6, portion_foreground=portion_foreground,
                            pairs=[(1, 2), (3, 4)],
                            morph=True
                            # pairs=[(4, 2)]
                            )



#
# image = Image.open(image_path)
# image = np.asarray(image.convert('RGB'))


