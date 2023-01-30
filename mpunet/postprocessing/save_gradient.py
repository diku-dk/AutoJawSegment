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


def sobel3D(img_data, sigma=3):
    # import skimage.filters.sobel as sk_sobel
    if sigma > 0:
        img_data = ndimage.gaussian_filter(img_data, sigma=sigma)

    dx = ndimage.sobel(img_data, 0)  # x derivative
    dy = ndimage.sobel(img_data, 1)  # y derivative
    dz = ndimage.sobel(img_data, 2)  # z derivative
    # dx = sk_sobel(img_data, 0)
    img_data = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    # img_data = np.abs(dx) + np.abs(dy) + np.abs(dz)
    img_data = img_data.astype(np.float32)

    theta = [np.arctan2(dy, dx), np.arctan2(dz, dx), np.arctan2(dz, dy)]

    return img_data, theta


if __name__ == '__main__':

    root_path = '/Users/px/Downloads/predictions_no_arg_w_softmax_teeth'

    pred_grads_path = os.path.join(root_path, 'pred_grads')

    binary_path = os.path.join(root_path, 'binary_path')

    if not os.path.exists(pred_grads_path):
        os.mkdir(pred_grads_path)

    if not os.path.exists(binary_path):
        os.mkdir(binary_path)

    for i in os.listdir(root_path):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue

        img_path = os.path.join(root_path, i)

        save_path = os.path.join(pred_grads_path, i)

        img_func = nib.load(img_path)
        affine = img_func.affine
        img = img_func.get_fdata()
        img = np.squeeze(img)
        # img = img[...,2]
        #
        # img = img == 2

        # img /= 6

        # grad_magnitude, _ = sobel3D(img, sigma=3)

        grad_magnitude = np.gradient(img)
        grad_magnitude = np.array(grad_magnitude) ** 2
        grad_magnitude = np.sum(grad_magnitude, axis=0)
        grad_magnitude = np.sqrt(grad_magnitude)
        ni_img = nib.Nifti1Image(grad_magnitude.astype(np.float32)
                                 , affine)

        nib.save(ni_img, save_path)


        save_path = os.path.join(binary_path, i)
        binary_img = img > 0.5
        ni_img = nib.Nifti1Image(binary_img.astype(np.uint8)
                                 , affine)
        nib.save(ni_img, save_path)
