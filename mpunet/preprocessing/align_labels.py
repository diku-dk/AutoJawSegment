import os
from shutil import copyfile
import numpy as np
from utils import *

from simple_thresholding import simple_thresholding



def align_labels(file_path, key_map):
    img_func = nib.load(file_path)
    affine = img_func.affine
    img_data = img_func.get_fdata()

    img_data_new = np.copy(img_data).astype(np.uint8)



    for i in key_map.keys():
        img_data_new[img_data == i] = key_map[i]

    img_data_new = img_data_new.astype(np.uint8)

    ni_img = nib.Nifti1Image(img_data_new, affine)

    nib.save(ni_img, file_path)

    # a = np.random.randn(3,3)
    # b = np.zeros((3,3)).astype(np.bool)
    # b[1, 1] = True
    # a[b] = 2

    return



if __name__ == '__main__':

    import os

    # root = '/Users/px/Downloads/training_same_FOV/aug'
    # for i in os.listdir(root):
    #     if i.startswith('.'): continue
    #     sub = os.path.join(root, i)
    #     for j in os.listdir(sub):
    #         if j.startswith('.'): continue
    #
    #         before = os.path.join(sub, j)
    #         after = before.replace('0.', '0_')
    #         os.rename(before, after)


    root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/testdata0902/images_no_thresh'

    root = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecAll/train/labels'


    maps = {}
    maps['Patient_4_0.15mm_fixed_dim.nii.gz'] = {3: 4, 4: 3}

    for i in os.listdir(root):
        if i.startswith('.'): continue
        if i not in maps.keys(): continue

        path = os.path.join(root, i)
        align_labels(file_path=path,
                     key_map=maps[i])
