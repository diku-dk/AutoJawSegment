import os.path

import nibabel as nib
import nibabel.processing
from skimage.transform import rescale

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import color, data, filters, graph, measure, morphology
import pyvista

#
# path = '/Volumes/T7/kidneys/nii_images/images/CT_rat15_kidneyProc.nii'
# input_img = nib.load(path)
# img = input_img.get_fdata()
# affine = input_img.affine
# img = nib.Nifti1Image(img, affine)
# output_path = '/Volumes/T7/kidneys/nii_images/images/CT_rat15_kidneyProc.nii.gz'
# nib.save(img, output_path)

path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/vessels/train/labels/cropped_RAT_03_reshaped_all_c.nii.gz'
path = '/Volumes/T7/kidneys/nii_images/labels/RAT_03_vessels_segmented.nii.gz'

path = '/Volumes/T7/kidneys/nii_images/labels/Rat_37_zoom_seg_clean.nii.gz'

input_img = nib.load(path)
img = input_img.get_fdata()
affine = input_img.affine

img = img.astype(np.uint8)

out_root = os.path.dirname(path)

res = (img == np.max(img)).astype(np.uint8)
res = nib.Nifti1Image(res, affine)
output_path = os.path.join(out_root, f'Rat_37_zoom_seg_clean_254.nii.gz')
nib.save(res, output_path)

for i in np.unique(img):
    res = (img == i).astype(np.uint8)
    res = nib.Nifti1Image(res, affine)
    output_path = os.path.join(out_root, f'cropped_RAT_03_reshaped_{i}.nii.gz')
    nib.save(res, output_path)

def labelimg_to_points(img, save_path='turning254.vtk'):
    coordinates = np.where(img != 0)
    coordinates = np.array(coordinates)
    coordinates = coordinates.T
    mesh = pyvista.PolyData(coordinates,
                            )
    mesh.save(save_path)

    # mesh = pyvista.read(pt_file)
    # pts_coords = mesh.points