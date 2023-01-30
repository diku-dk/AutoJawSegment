import os
from shutil import copyfile
import numpy as np
from utils import *

from simple_thresholding import simple_thresholding

def separate_imgaes_labels(root_path):


    # after_root = os.path.join(root_path,
    #                           'jaw_015_cropped')

    after_root = root_path

    print(f'after: {after_root}')
    images_root = os.path.join(after_root, 'images')
    labels_root = os.path.join(after_root, 'labels')

    if not os.path.exists(after_root):
        os.mkdir(after_root)

    if not os.path.exists(images_root):
        os.mkdir(images_root)

    if not os.path.exists(labels_root):
        os.mkdir(labels_root)

    file_names = os.listdir(root_path)

    for file in file_names:
        file_dir = os.path.join(root_path, file)
        if not os.path.isdir(file_dir) or 'images' in file or 'labels' in file:
            print(f'skip {file}')
            continue

        n = file[8:]

        image_path_before = os.path.join(file_dir, f'CBCT_scan_{n}.nii.gz')
        label_path_before = os.path.join(file_dir, f'Segmentation-label_{n}.nii.gz')

        image_path_after = os.path.join(images_root, file+'.nii.gz')
        label_path_after = os.path.join(labels_root, file+'.nii.gz')

        copyfile(image_path_before, image_path_after)
        copyfile(label_path_before, label_path_after)

def expand_dims_images(colab=False, root_path='', ignore_label=False):

    images_root = os.path.join(root_path, 'images')
    labels_root = os.path.join(root_path, 'labels')

    file_names = os.listdir(images_root)

    for file in file_names:
        if file.startswith('.'): continue

        image_file_dir = os.path.join(images_root, file)
        label_file_dir = os.path.join(labels_root, file)

        image_file_dir =  '/Users/px/Downloads/CBCT_scan_1_0.15mm.nii'

        img = loadNii(image_file_dir).astype(np.float32)
        img_func = nib.load(image_file_dir)

        if img.shape[-1] != 1:
            img = np.expand_dims(img, axis=-1)
            ni_img = nib.Nifti1Image(img, img_func.affine)
            nib.save(ni_img, image_file_dir)

        if ignore_label: break

        label = loadNii(label_file_dir).astype(np.uint8)

        if label.shape[-1] != 1:
            label_func = nib.load(label_file_dir)
            label = np.expand_dims(label, axis=-1)
            ni_label = nib.Nifti1Image(label, label_func.affine)
            nib.save(ni_label, label_file_dir)


def expand_dims_images2(colab=False, images_root='', ignore_label=False):


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
    root_path = '/Users/px/Downloads/jawDecLow'

    separate_imgaes_labels(root_path)

    root = '/Users/px/Downloads/OneDrive_1_10-12-2021'

    expand_dims_images(root_path=root, ignore_label=False)



    expand_dims_images2(images_root=root, ignore_label=False)

    expand_dims_images(root_path=root, ignore_label=False)

    images_root = os.path.join(root, 'images')

    for i in os.listdir(images_root):
        if i.startswith('.'): continue
        path = os.path.join(images_root, i)
        simple_thresholding(file_path=path, save_path=path,
                            thresh=0, to_label=False)

    correct_label = False
