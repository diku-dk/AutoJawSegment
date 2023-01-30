import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import skimage
from skimage import filters
from PIL import Image

from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_opening)
from skimage.morphology import (erosion, dilation, opening, closing, binary_closing,
                                binary_dilation, binary_erosion,
                                area_closing,
                                white_tophat, remove_small_holes)

from scipy.ndimage import distance_transform_edt

from mpunet.postprocessing.connected_component_3D import *

from scipy import ndimage as ndi
import matplotlib.pyplot as plt

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage import data
from skimage.filters import rank
from skimage.util import img_as_ubyte


def reset_reanrange(labels, thresh=0.1):
    res = np.zeros(labels.shape)
    num_forge = np.sum(labels != 0)
    print(f'total num of forg = {num_forge}')

    labels_indices = sorted(np.unique(labels))
    re_aranged_ind = 1
    for i in labels_indices[1:]:
        cur_sum = np.sum(labels == i)

        print(f'total num of cur_sum = {cur_sum}')

        if cur_sum > num_forge * thresh:
            res[labels == i] = re_aranged_ind
            re_aranged_ind += 1

    return res


def watershed_w_surface(pred_img, grad_img=None, surf_img=None):

    if grad_img is None:
        grad_img = np.gradient(pred_img)
        grad_img = np.array(grad_img) ** 2
        grad_img = np.sum(grad_img, axis=0)
        grad_img = np.sqrt(grad_img)

        # grad_img = ndi.gaussian_gradient_magnitude(pred_img, sigma=2)

    binary_img_fg = pred_img > 0.7
    # binary_img_fg = connected_component_3D(binary_img_fg, connectivity=26, portion_foreground=0.01)

    binary_img_bg = pred_img > 0.3

    del pred_img

    opening = binary_opening(binary_img_fg, ball(5))
    closing = binary_closing(binary_img_bg, ball(3))
    dilate_close = binary_dilation(closing, ball(3))

    sure_fg = binary_erosion(opening, ball(3))

    sure_fg = connected_component_3D(sure_fg, connectivity=26, portion_foreground=0.01) > 0


    ni_img = nib.Nifti1Image(sure_fg.astype(np.uint8)
                             , affine)
    nib.save(ni_img, 'sure_fg.nii.gz')

    # sure_bg = 1 - dilate_close  # sure background area

    distances = distance_transform_edt(1 - dilate_close)

    sure_bg = watershed(distances, watershed_line=True)
    sure_bg = sure_bg == 0

    ni_img = nib.Nifti1Image(sure_bg.astype(np.uint8)
                             , affine)
    nib.save(ni_img, 'sure_bg.nii.gz')

    # from skimage.morphology import skeletonize
    # sure_bg = skeletonize(sure_bg)

    unknown = 1 - np.logical_or(sure_bg, sure_fg)

    markers, num_labels = label(sure_fg)
    unique_class_before = np.unique(markers)

    markers = reset_reanrange(markers, 0.01)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    markers[sure_bg == 1] = 1
    # Now, mark the region of unknown with zero
    markers[unknown == 1] = 0

    # combined = grad_img * 0.35 + surf_img * 0.65


    res = watershed(grad_img, markers) - 1
    #
    # ni_img = nib.Nifti1Image(res.astype(np.float32)
    #                          , affine)
    # nib.save(ni_img, 'watershed.nii.gz')


    return res

def test():
    image = img_as_ubyte(data.eagle())

    # denoise image
    denoised = rank.median(image, disk(2))

    # find continuous region (low gradient -
    # where less than 10 for this image) --> markers
    # disk(5) is used here to get a more smooth image
    markers = rank.gradient(denoised, disk(5)) < 10
    markers = ndi.label(markers)[0]

    # local gradient (disk(2) is used to keep edges thin)
    gradient = rank.gradient(denoised, disk(2))

    # process the watershed
    labels = watershed(gradient, ((markers == 487) + 1).astype(np.int32))

    plt.imshow(labels)

    Mf = binary_opening(img, ball(5))
    Mf = binary_erosion(Mf, ball(2))
    dialted = binary_erosion(img, ball(3))


if __name__ == '__main__':

    root_path = '/Users/px/Downloads/predictions_no_arg/nii_files_task_0'
    pred_grads_root = os.path.join(root_path, 'pred_grads')

    # surface_root = os.path.join(root_path, 'surface')

    save_root = os.path.join(root_path, 'watershed')

    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for i in os.listdir(root_path):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue

        # surf_path = os.path.join(surface_root, i)
        pred_grads_path = os.path.join(pred_grads_root, i)
        pred_path = os.path.join(root_path, i)

        img_func = nib.load(pred_grads_path)
        img_grad = img_func.get_fdata().astype(np.float32)
        img_grad = np.squeeze(img_grad)

        # pred_path = '/Users/px/Downloads/Patient_25_0_PRED.nii.gz'

        img_func = nib.load(pred_path)
        affine = img_func.affine
        pred_img = img_func.get_fdata().astype(np.float32)
        pred_img = np.squeeze(pred_img)

        # img_func = nib.load(surf_path)
        # affine = img_func.affine
        # surf_img = img_func.get_fdata().astype(np.float32)
        # surf = np.squeeze(surf == 2)

        save_path = os.path.join(save_root, i)
        res = watershed_w_surface(pred_img, grad_img=img_grad, surf_img=None)

        ni_img = nib.Nifti1Image(res.astype(np.float32)
                                 , affine)
        nib.save(ni_img, save_path)



def OTSU_segment(img, th_begin=0, th_end=256, interval=1):
    max_g = 0
    thresh_opt = 0

    for threshold in range(th_begin, th_end, interval):
        fore_mask = img > threshold
        back_mask = img <= threshold

        fore_num = np.sum(fore_mask)
        back_num = np.sum(back_mask)

        if 0 == fore_num or 0 == back_num:
            continue

        w0 = float(fore_num) / img.size
        u0 = float(np.sum(img * fore_mask)) / fore_num
        w1 = float(back_num) / img.size
        u1 = float(np.sum(img * back_mask)) / back_num
        # intra-class variance
        g = w0 * w1 * (u0 - u1) ** 2

        if g > max_g:
            max_g = g
            thresh_opt = threshold
        if threshold % 10 == 0:
            print(f'current best thresh = {thresh_opt}')
    return thresh_opt


