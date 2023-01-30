import shutil

import numpy as np


from mpunet.postprocessing.connected_component_3D import connected_component_3D

from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.morphology import (ball, binary_opening, binary_erosion)

from background_compute import get_distance_minus
from foreground_compute import ccd_separate
from scipy.ndimage import gaussian_gradient_magnitude
import os
import nibabel as nib
import sys


class Watershed:
    def __init__(self, root, true_foreground='true_foreground', true_background='true_background',
                 ignore_computed=True):
        self.root = root
        self.true_fg_root = os.path.join(root, true_foreground)
        self.true_bg_root = os.path.join(root, true_background)
        self.prob_img_root = os.path.join(root, 'images')
        self.grad_img_root = os.path.join(root, 'grad')
        self.watershed_root = os.path.join(root, 'watershed')
        self.combined_root = os.path.join(root, 'combined')
        self.combined_binary_root = os.path.join(root, 'combined_binary')

        self.prediction_root = os.path.join(root, 'binary')

        self.ignore_computed = ignore_computed

        for i in [self.true_fg_root, self.true_bg_root, self.prob_img_root, self.grad_img_root, self.watershed_root]:
            if not os.path.exists(i):
                os.makedirs(i)

    def compute_foreground(self, thresh=0.6, radius=3, erosion_radius=1, portion_foreground=0.05):
        print('computing foreground...')

        for i in os.listdir(self.prob_img_root):
            if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')): continue

            path = os.path.join(self.prob_img_root, i)
            save_path = os.path.join(self.true_fg_root, i)

            if self.ignore_computed and os.path.exists(save_path): continue

            img = nib.load(path)
            img_data = img.get_fdata()
            img_data = np.squeeze(img_data)
            res = img_data > thresh
            res = res.astype(np.uint8)
            res = binary_opening(res, ball(radius))
            res = binary_erosion(res, ball(erosion_radius))

            res = ccd_separate(res, portion_foreground=portion_foreground)
            res = res.astype(np.uint8)

            # save_path = os.path.join(self.true_fg_root, str(len(np.unique(res))) + i)

            save_path = os.path.join(self.true_fg_root, i)

            nib.save(nib.Nifti1Image(res, img.affine), save_path)

    def compute_background(self, dist_thresh=1, sum_thresh=20, prob_thres=0.5):
        print('computing background...')

        for i in os.listdir(self.true_fg_root):
            if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')): continue

            save_root = self.true_bg_root
            save_path = os.path.join(save_root, i)

            if self.ignore_computed and os.path.exists(save_path): continue

            img = nib.load(os.path.join(self.true_fg_root, i))

            # prob_img = nib.load(os.path.join(root, i)).get_fdata()
            # binary_img_bg = binary_img_bg > prob_thresh
            # closing = binary_closing(binary_img_bg, ball(3))
            # dilate_close = binary_dilation(closing, ball(3))
            # sure_bg = 1 - dilate_close  # sure background area

            img_data = img.get_fdata()

            img_data = img_data.astype(np.uint8)

            res = get_distance_minus(img_data, thresh=dist_thresh, sum_thresh=sum_thresh)

            nib.save(nib.Nifti1Image(res, img.affine), save_path)

    def compute_grad(self, sigma=2):
        print('computing final gradient...')

        for i in os.listdir(self.prob_img_root):
            if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
                continue

            save_root = self.grad_img_root
            save_path = os.path.join(save_root, i)

            if self.ignore_computed and os.path.exists(save_path): continue

            pred_path = os.path.join(self.prob_img_root, i)
            img_func = nib.load(pred_path)
            pred_img = img_func.get_fdata().astype(np.float32)
            pred_img = np.squeeze(pred_img)

            affine = img_func.affine

            rad_img = gaussian_gradient_magnitude(pred_img, sigma=sigma)

            rad_img = rad_img.astype(np.float32)

            ni_img = nib.Nifti1Image(rad_img.astype(np.float32)
                                     , affine)

            nib.save(ni_img, save_path)


    def watershed(self, grad=True):

        print('computing final watershed...')

        img_path = self.grad_img_root if grad else self.prob_img_root

        for i in os.listdir(img_path):
            if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
                continue

            save_path = os.path.join(self.watershed_root, i)

            if self.ignore_computed and os.path.exists(save_path): continue

            pred_path = os.path.join(img_path, i)
            bg_path = os.path.join(self.true_bg_root, i)
            fg_path = os.path.join(self.true_fg_root, i)

            img_func = nib.load(pred_path)
            pred_img = img_func.get_fdata().astype(np.float32)
            pred_img = np.squeeze(pred_img)

            affine = img_func.affine

            sure_fg = np.squeeze(nib.load(fg_path).get_fdata().astype(np.uint8))
            sure_bg = np.squeeze(nib.load(bg_path).get_fdata().astype(np.uint8))

            sure_bg = sure_bg * (sure_fg == 0)

            unknown = 1 - np.logical_or(sure_bg, sure_fg)

            # markers, num_labels = label(sure_fg)
            markers = sure_fg

            # unique_class_before = np.unique(markers)
            # markers = reset_reanrange(markers, 0.01)

            # Add one to all labels so that sure background is not 0, but 1
            markers = markers + 1
            markers[sure_bg == 1] = 1
            # Now, mark the region of unknown with zero
            markers[unknown == 1] = 0

            # pred_img
            res = watershed(pred_img, markers) - 1
            #
            ni_img = nib.Nifti1Image(res.astype(np.float32)
                                     , affine)

            nib.save(ni_img, save_path)

    def combine_res(self):


        file_names = [i for i in os.listdir(self.watershed_root) if
                      (not i.startswith('.') and (i.endswith('nii') or i.endswith('gz')))]
        if not os.path.exists(self.combined_root):
            os.mkdir(self.combined_root)

        if not os.path.exists(self.combined_binary_root):
            os.mkdir(self.combined_binary_root)

        for i in file_names:

            water = os.path.join(self.watershed_root, i)
            water = nib.load(water).get_fdata().astype(np.uint8)

            predictions = os.path.join(self.prediction_root, i)
            predictions = nib.load(predictions)
            affine = predictions.affine
            predictions = predictions.get_fdata().astype(np.uint8)

            bone = predictions == 1
            # upper_bone = predictions == 4

            res = np.zeros(predictions.shape, dtype=np.uint8)

            res[bone] = 1
            # res[upper_bone] = 2

            res[water > 0] = water[water > 0] + 1

            save_name = os.path.join(self.combined_root, i)

            nib.save(nib.Nifti1Image(res.astype(np.uint8), affine), save_name)

            res_binary = np.zeros(predictions.shape, dtype=np.uint8)
            res_binary[bone] = 1

            res_binary[water > 0] = 2

            save_name = os.path.join(self.combined_binary_root, i)

            nib.save(nib.Nifti1Image(res_binary.astype(np.uint8), affine), save_name)


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

if __name__ == '__main__':

    unet_thresh_val = 0.8
    open = 5
    erosion = 3

    d_min = 1
    d_max = 35

    # root = sys.argv[1]
    root = '/Users/px/Downloads/AllRes/newGap'

    # for thresh in [0.8, 0.85]:
    #     for open in np.arange(2, 6):
    #         for erosion in np.arange(2, 5):
    #             if open + erosion < 5:
    #                 continue
    #             else:
    #                 true_foreground = f'true_foreground{open}_{erosion}_{thresh}'
    #                 water = Watershed(root=root, true_foreground=true_foreground, ignore_computed=True)
    #                 water.compute_foreground(thresh=thresh, radius=open, erosion_radius=erosion, portion_foreground=0.01)

    true_foreground = f'true_foreground{open}_{erosion}_{unet_thresh_val}'
    water = Watershed(root=root, true_foreground=true_foreground, ignore_computed=True, true_background='true_background')
    # water.compute_foreground(thresh=unet_thresh_val, radius=open, erosion_radius=erosion)
    water.compute_background(dist_thresh=d_min, sum_thresh=d_max)
    water.compute_grad()
    water.watershed(grad=True)
    water.combine_res()