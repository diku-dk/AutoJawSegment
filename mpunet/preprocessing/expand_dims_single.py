import os
from shutil import copyfile
import numpy as np
from utils import *

from simple_thresholding import simple_thresholding

def expand_dims_images(images_root=''):

    file_names = os.listdir(images_root)

    for file in file_names:
        if file.startswith('.'): continue

        image_file_dir = os.path.join(images_root, file)
        # save_file_dir = os.path.join(images_root, file.split('-')[0]+'.nii')
        img = loadNii(image_file_dir).astype(np.float32)
        img_func = nib.load(image_file_dir)

        if img.shape[-1] != 1:
            img = np.expand_dims(img, axis=-1)
            ni_img = nib.Nifti1Image(img, img_func.affine)
            print(f'saving for {file}')
            nib.save(ni_img, image_file_dir)


if __name__ == '__main__':
    root = '/Users/px/Downloads/our_images_cropped'

    expand_dims_images(images_root=root)
