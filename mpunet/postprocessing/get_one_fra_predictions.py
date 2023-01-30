import numpy as np
from skimage.io import imshow
# from skimage.measure import label

from scipy.ndimage import label, generate_binary_structure
from scipy import ndimage

from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import os
from mpunet.postprocessing.connected_component_3D import connected_component_3D


if __name__ == '__main__':

    img_path = '/Users/px/Downloads/predictions_no_arg/Patient_11_0_PRED.nii.gz'
    save_path = '/Users/px/Downloads/predictions_no_arg/Patient_11_0_PRED_grad.nii.gz'
