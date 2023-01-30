import numpy as np
from skimage.io import imshow
# from skimage.measure import label

from scipy.ndimage import label, generate_binary_structure

from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import os

if __name__ == '__main__':

    first_root = '/Users/px/Downloads/OneDrive_1_16-11-2022/weight_map_teeth'
    second_root = '/Users/px/Downloads/OneDrive_1_16-11-2022/weight_maps_gap'
    save_root = '/Users/px/Downloads/OneDrive_1_16-11-2022/weight_map_final'

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for i in os.listdir(first_root):
        if i.startswith('.') or not (i.endswith('.nii') or i.endswith('.gz')): continue

        path = os.path.join(first_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img_data = img_func.get_fdata()
        img_data = np.squeeze(img_data).astype(np.float32)

        path_2 = os.path.join(second_root, i)
        img_data_2 = np.squeeze(nib.load(path_2).get_fdata()).astype(np.float32)

        res = (img_data + img_data_2).astype(np.float32)

        ni_img = nib.Nifti1Image(res, affine)
        save_path = os.path.join(save_root, i)
        nib.save(ni_img, save_path)
