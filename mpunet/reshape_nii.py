import numpy as np
import skimage.transform as skTrans
from nilearn.image import resample_img

import nibabel as nib


filename = '/Users/px/GoogleDrive/MultiPlanarUNet/Patient_15/CBCT_scan.nii'
img = nib.load(filename)

downsampled_nii = resample_img(img, target_affine=np.eye(3)*0.5, interpolation='nearest')

nib.save(downsampled_nii, ''.join(filename.split('.')[:-1]) + str('_down.nii'))

affine = img.affine

data = img.get_fdata()

input = skTrans.resize(input, dim,
                       order=1, preserve_range=True)
