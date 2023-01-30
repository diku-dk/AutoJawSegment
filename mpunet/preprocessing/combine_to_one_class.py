import nibabel as nib
import numpy as np
import os

def combine_to_one_class(file_path, save_path):

    file_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jaw0920_w_gap/images/CBCT_scan_4_0.15mm.nii'

    file_path = '/Users/px/Downloads/Patient_4_prime/CBCT_scan_4_0.15mm.nii'

    img_func = nib.load(file_path)
    affine = img_func.affine
    img_data = img_func.get_fdata()

    img_data = (img_data > 0).astype('uint8')

    ni_img = nib.Nifti1Image(img_data, affine)

    nib.save(ni_img, save_path)


def combine_certain_classes(class_dict, file_path, save_path):
    img_func = nib.load(file_path)
    affine = img_func.affine
    img_data = img_func.get_fdata()

    for (i, j), target in class_dict.items():
        img_data[np.logical_or(img_data == i, img_data == j)] = target

    img_data = img_data.astype('uint8')

    ni_img = nib.Nifti1Image(img_data, affine)

    nib.save(ni_img, save_path)

if __name__ == '__main__':

    root_path = '/Users/px/Downloads/AllRes/NewTest/not included'

    labels_path = os.path.join(root_path, 'labels')

    res_path = os.path.join(root_path, 'labels_merge_up_low')

    if not os.path.exists(res_path):
        os.mkdir(res_path)

    class_dict = {(1, 4): 1, (2, 3): 2}



    for i in os.listdir(labels_path):
        if i.startswith('.'):
            continue
        path = os.path.join(labels_path, i)
        save_path = os.path.join(res_path, i)
        combine_certain_classes(class_dict, path, save_path)