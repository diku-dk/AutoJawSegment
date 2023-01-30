import sys, os
import numpy as np
import nibabel as nib

import czifile
# from aicsimageio import AICSImage

import SimpleITK as sitk
import scipy

root = '/Volumes/T7/kidneys/tiffs/images'
for i in os.listdir(root):
    if i.endswith(".tiff") and (not i.startswith('.')):

        before = os.path.join(root, i)
        after = os.path.join(root, i.replace('.tiff', '.nii.gz'))
        img = sitk.ReadImage(before)
        sitk.WriteImage(img, after)
