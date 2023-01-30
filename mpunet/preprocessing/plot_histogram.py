import sys

import numpy as np
from skimage.io import imshow
# from skimage.measure import label

from scipy.ndimage import label, generate_binary_structure

from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import nibabel as nib
import os

from mpunet.preprocessing.expand_gaps import binary_erosion_label
#
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_opening)
from skimage.morphology import (erosion, dilation, opening, closing, binary_closing,
                                binary_dilation, binary_erosion, binary_erosion,
                                area_closing,
                                white_tophat, remove_small_holes)

from skimage import exposure



if __name__ == '__main__':



    images_root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jawDecAll/images'

    plot_root = os.path.join(os.path.dirname(images_root), 'hist')

    save_root = os.path.join(os.path.dirname(images_root), 'hist_normalized')


    if not os.path.exists(plot_root):
        os.mkdir(plot_root)
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for i in os.listdir(images_root):
        if i.startswith('.'): continue

        path = os.path.join(images_root, i)
        img_func = nib.load(path)
        img_data = img_func.get_fdata().astype(np.float32)
        img_data = np.squeeze(img_data)

        # img_data = exposure.equalize_hist(img_data)

        up_bound = np.quantile(img_data, 0.99)
        low_bound = np.quantile(img_data, 0.01)
        ranges = (up_bound-low_bound)/12

        img_data[img_data > up_bound] = up_bound + (img_data[img_data > up_bound] - up_bound)/(
                np.max(img_data)-np.min(img_data)) * ranges

        img_data[img_data < low_bound] = (img_data[img_data < low_bound]-np.min(img_data))/(
                low_bound-np.min(img_data)) * ranges + low_bound

        # quants = np.array([np.quantile(img_data, i)
        #                    # for i in np.arange(0.1, 1, 0.1)
        #                    for i in (0.005, 0.05, 0.95, 0.995)
        #                    ])
        #
        # quants = np.round(quants).astype(np.int16)

        up_bound = np.quantile(img_data, 0.99)
        low_bound = np.quantile(img_data, 0.01)
        quants = np.round([low_bound, up_bound]).astype(np.int16)

        plt.figure()
        plt.hist(np.reshape(img_data, -1), 50, density=True, )
        plt.title(quants)
        plt.savefig(os.path.join(plot_root, f'{i}.png'))

        save_path = os.path.join(save_root, i)
        nib.save(nib.Nifti1Image(img_data, img_func.affine), save_path)
