import nibabel as nib
import numpy as np
import os


def filter_some(file_path, save_path):
    img_func = nib.load(file_path)
    affine = img_func.affine
    img_data = img_func.get_fdata()

    assert img_data.min() >= 0 and img_data.max() <= 4

    img_data[img_data > 2] = 0

    # img_data[~np.isin(img_data, np.arange(5))] = 0

    img_data = img_data.astype('uint8')

    ni_img = nib.Nifti1Image(img_data, affine)

    nib.save(ni_img, save_path)



if __name__ == '__main__':

    root_path = '/Users/px/Downloads/jaw0920_w_gap'

    root_path = '/Users/px/Downloads/training_same_FOV/aug'

    root_path = '/Users/px/Downloads/jaw_0_115_cropped/aug'

    labels_path = os.path.join(root_path, 'labels')

    binary_path = os.path.join(root_path, 'labels_filtered')

    if not os.path.exists(binary_path):
        os.mkdir(binary_path)

    for i in os.listdir(labels_path):
        if i.startswith('.'):
            continue
        path = os.path.join(labels_path, i)

        save_path = os.path.join(binary_path, i)

        filter_some(file_path=path, save_path=save_path)
