import numpy as np
import tensorflow as tf
from mpunet.evaluate.metrics import *
from compute_metrics import save_gap_regions
from scipy.ndimage import label, generate_binary_structure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import os
from mpunet.preprocessing.extract_surface import extract_surface_morph
import pandas as pd


def remap_binary_gt(img):
    img[np.logical_or(img==1, img==4)] = 1
    img[np.logical_or(img==2, img==3)] = 2
    return img

def remap_binary_indv_tooth(img):
    img[np.logical_or(img==1, img==2)] = 1
    img[img>=3] = 2
    return img




if __name__ == '__main__':


    label_root = '/Users/px/Downloads/jawVal/label'
    water_root = '/Users/px/Downloads/jawVal/watershed'
    orig_root = '/Users/px/Downloads/jawVal/orig'
    weight_root = '/Users/px/Downloads/jawVal/weight_map_morph_teeth'


    assd = 0
    hausdorff_single = 0

    labels_path = [i for i in os.listdir(label_root)
                   if not (i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')))]

    orig_path = [i for i in os.listdir(orig_root)
                   if not (i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')))]

    water_path = [i for i in os.listdir(water_root)
                   if not (i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')))]

    weight_path = [i for i in os.listdir(weight_root)
                  if not (i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')))]

    metrics = ['dice_final', 'dice_gap_final', 'assd_final', 'hausdorff_final']

    for m in metrics:
        exec(f'{m} = []')

    for i, j, k, l in zip(sorted(labels_path), sorted(orig_path), sorted(water_path), sorted(weight_path)):

        # print(f'working on {i}')

        label_path = os.path.join(label_root, i)
        orig_path = os.path.join(orig_root, j)
        water_path = os.path.join(water_root, k)
        weight_path = os.path.join(weight_root, k)

        img_func = nib.load(label_path)
        affine_1 = img_func.affine
        label = img_func.get_fdata().astype(np.uint8)
        label = np.squeeze(label)
        # img = extract_surface_morph(img, radius=3, bg_val=0)

        img_func = nib.load(orig_path)
        affine_2 = img_func.affine
        pred_orig = img_func.get_fdata().astype(np.uint8)
        pred_orig = np.squeeze(pred_orig)

        img_func = nib.load(water_path)
        pred_water = img_func.get_fdata().astype(np.uint8)
        pred_water = np.squeeze(pred_water)
        # img2 = extract_surface_morph(img2, radius=3, bg_val=0)

        weight = np.squeeze(nib.load(weight_path).get_fdata().astype(np.float32))

        if label.shape != pred_orig.shape:
            print(f'cannot work on label {label.shape} and pred {pred_orig.shape}')
            continue

        pred_orig = np.logical_or(pred_orig == 2, pred_orig == 3)
        label = np.logical_or(label == 2, label == 3)
        pred_water = pred_water > 0

        dices_orig = dice(label, pred_orig)
        dices_water = dice(label, pred_water)

        acc_orig = np.sum((label == pred_orig)) / np.prod(label.shape)
        acc_water = np.sum((label == pred_water)) / np.prod(label.shape)

        print(f' orig dice {dices_orig}, dices_water {dices_water}')
        # print(f' orig acc {acc_orig}, acc_water {acc_water}')

        continue

        # mask = cal_gap(label, pred_orig, dist=10,
        #                use_predict=False
        #                )
        mask = weight > 0.1
        path = '/Users/px/Downloads/jawVal/gap'

        save_gap_regions(path, label, pred_orig, mask, i, affine_1)

        dice_gap_orig = dice_all(label[mask], pred_orig[mask],
                              # ignore_zero=False,
                              )

        dice_gap_water = dice_all(label[mask], pred_water[mask],
                              # ignore_zero=False,
                              )
        print(f' orig dice {dice_gap_orig}, dices_water {dice_gap_water}')
