import sys

import numpy as np
from skimage.io import imshow
# from skimage.measure import label

from scipy.ndimage import label, generate_binary_structure

from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import os


if __name__ == '__main__':

    weight_root = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecLabelsAll/weight_map_revert_combined'

    for i in os.listdir(weight_root):
        if i.startswith('.'): continue

        path = os.path.join(weight_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img_data = img_func.get_fdata()
        img_data = np.squeeze(img_data)

        img_data[np.isclose(img_data, -0.9)] = -1
        img_data = img_data.astype(np.float32)

        ni_img = nib.Nifti1Image(img_data, affine)
        save_path = os.path.join(weight_root, i)
        nib.save(ni_img, save_path)
