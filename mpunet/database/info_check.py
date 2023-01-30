import sys

import numpy as np
import os
from scipy.ndimage import distance_transform_edt
import nibabel as nib

def get_pix_dim(nii_image):
    return nii_image.header["pixdim"][1:4]

if __name__ == '__main__':
    root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/hip_all'
    dims = []
    pix_dims = []
    for i in os.listdir(root):
        if i.startswith('.') or not (i.endswith('.gz') or i.endswith('.nii')):
            continue

        path = os.path.join(root, i)
        a = nib.load(path)
        shape = np.array(a.shape)
        if shape[-1] == 1:
            shape = shape[:-1]
        if shape[0] == 1:
            shape = shape[1:]
        print(i, shape)
        dims.append(shape)
        pix_dims.append(get_pix_dim(a))

    dims = np.array(dims)
    pix_dims = np.array(pix_dims)

    print(np.mean(dims, axis=0))
    print(np.std(dims, axis=0))

    print(np.mean(pix_dims, axis=0))
    print(np.std(pix_dims, axis=0))