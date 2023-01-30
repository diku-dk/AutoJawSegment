
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import os

from mpunet.postprocessing.connected_component_separator import connected_component_separate_3D
import numpy as np
# from mpunet.postprocessing.morphologies import binary_closing_label
# from mpunet.postprocessing.morphologies import binary_opening_label

from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_opening)

def binary_opening_label(img, radius=1,  bg_val=0):

    res = np.zeros(img.shape, dtype=np.uint8)

    footprint = ball(radius)

    for label in sorted(np.unique(img)):
        if label == bg_val:
            continue

        img_cur = (img == label).astype(np.uint8)

        labels_out = binary_opening(img_cur, footprint)

        # labels_out = binary_erosion(labels_out, footprint)

        res += (int(label) * labels_out).astype(np.uint8)

        res[res > int(label)] = 0

        print(np.unique(res))

    res[res > np.max(img)] = 0

    res = res.astype('uint8')

    return res

def save_only_tooth(img):
    img[img == 4] = 0
    img[img == 1] = 0

    return img

if __name__ == '__main__':

    root_path = '../../data_folder/jawDecLabelsAll/labels'
    labels_only_tooth_path = os.path.join(os.path.dirname(root_path), 'labels_only_tooth')
    #
    ccd_path = os.path.join(os.path.dirname(root_path), 'labels_indiv_teeth')

    if not os.path.exists(labels_only_tooth_path):
        os.mkdir(labels_only_tooth_path)

    if not os.path.exists(ccd_path):
        os.mkdir(ccd_path)


    for i in os.listdir(root_path):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue

        img_path = os.path.join(root_path, i)

        save_path = os.path.join(labels_only_tooth_path, i)

        ccd_save_path = os.path.join(ccd_path, i)

        img_func = nib.load(img_path)
        affine = img_func.affine
        img = img_func.get_fdata().astype(np.uint8)

        img = save_only_tooth(img)

        img = binary_opening_label(img, radius=1)

        # img = np.squeeze(img)
        ni_img = nib.Nifti1Image(img.astype(np.uint8)
                                 , affine)

        nib.save(ni_img, save_path)

        res = connected_component_separate_3D(save_path, save_path=ccd_save_path,
                                              connectivity=6,
                                              portion_foreground=0.1,
                                              erosion=False,
                                              n_class=30)

