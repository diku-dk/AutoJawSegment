# from imio import load, save
#
# img = load.load_any("/path/to/file.tiff")
# save.to_nii(img, "/path/to/file.nii")

from PIL import Image
import nibabel as nib
import numpy as np
import imageio

in_path = "C:\\Users\\Administrator\\Downloads\\train_001_V01.tiff"
# tiff_data = Image.open(in_path)
tiff_data = imageio.volread(in_path)

tiff_data = tiff_data.astype(np.float32)

# tiff_data = tiff_data.astype(np.uint8)

affine = np.eye(4)

nifti_image = nib.load("C:\\Users\\Administrator\\Downloads\\selected_files_2024814_135535\\Rasmus_knee_sample_scan\\annotation_preprocessed_Femur_Tibia_384x384.nii")

in_path = "C:\\Users\\Administrator\\Downloads\\selected_files_2024814_135535\\Rasmus_knee_sample_scan\\prediction_martin_384x384_2.nii.gz"
nifti_image = nib.load(in_path)
nifti_image = nib.Nifti1Image(nifti_image.get_fdata().astype(np.uint8), affine=affine)
nib.save(nifti_image, in_path)

import SimpleITK as sitk

# Load the NIfTI file
in_path = ("C:\\Users\\Administrator\\Downloads\\selected_files_2024814_135535\\"
           "Rasmus_knee_sample_scan\\annotation_preprocessed_Femur_Tibia_384x384.nii")
image = sitk.ReadImage(in_path)
image_array = sitk.GetArrayFromImage(image)
image_array = np.moveaxis(image_array, 1, -1)
affine = np.eye(4)

nifti_image = nib.load(r"C:\Users\Administrator\PycharmProjects\AutoJawSegment\rasmus\train\images\original_image_preprocessed_384x384.nii")
nib.save(nifti_image, r"C:\Users\Administrator\PycharmProjects\AutoJawSegment\rasmus\train\images\original_image_preprocessed_384x384.nii.gz")


nifti_image = nib.Nifti1Image(image_array.astype(np.uint8), affine=affine)
nib.save(nifti_image, in_path)


out_path = ''.join(in_path.split(".")[:-1]) + '.nii.gz'
nib.save(nifti_image, out_path)