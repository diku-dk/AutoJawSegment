from scipy.ndimage import label, generate_binary_structure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as nib_process
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale
import os



def test():
    img_path = '/Users/px/Downloads/training_same_FOV/aug/images/smooth_08_0_3CBCT_scan_4_0_15mm_fov.nii.gz'
    img_path = '/Users/px/Downloads/training_same_FOV/aug/images/0_3CBCT_scan_4_0_15mm_fov.nii.gz'
    img_path = '/Users/px/Downloads/training_same_FOV/train/images/CBCT_scan_4_0.15mm_fov.nii.gz'


    save_name = '/Users/px/Downloads/training_same_FOV/aug/images/smooth_1_0_3CBCT_scan_4_0_15mm_fov.nii.gz'

    fwhm = 0.5
    n_dim = len(img.shape)
    # TODO: make sure time axis is last
    # Pad out fwhm from scalar, adding 0 for fourth etc (time etc) dimensions
    fwhm = np.asarray(fwhm)
    if fwhm.size == 1:
        fwhm_scalar = fwhm
        fwhm = np.zeros((n_dim,))
        fwhm[:3] = fwhm_scalar
    # Voxel sizes
    RZS = img.affine[:, :n_dim]
    vox = np.sqrt(np.sum(RZS ** 2, 0))
    # Smoothing in terms of voxels
    vox_fwhm = fwhm / vox

    SIGMA2FWHM = np.sqrt(8 * np.log(2))

    vox_sd = vox_fwhm / SIGMA2FWHM
    # Do the smoothing

    img = nib.load(img_path)
    img2 = nib_process.smooth_image(img, 0.5)
    data = img.get_fdata()

    # sigma = x / (vox * 2sqrt(2ln2))

    data = rescale(data, 0.5,
                   preserve_range=True)

    affine_func = img.affine

    data = gaussian_filter(data, sigma=0.5).astype(np.float32)

    data = data.astype(np.float32)

    img2 = nib.Nifti1Image(data, affine_func)

    nib.save(img2, save_name)
