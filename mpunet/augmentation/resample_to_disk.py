import os.path
import os
seed = 42  # for reproducibility


import nibabel as nib
import nibabel.processing
from skimage.transform import rescale
import sys, os
import numpy as np
import nibabel as nib
from matplotlib import pyplot, transforms
from matplotlib.pyplot import figure
import czifile
import matplotlib.pyplot as plt
from skimage import filters
import scipy
from aug_save_to_disk import *

input_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jawDecAll/images'
save_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jawDecAll/images_resampled'

if not os.path.exists(save_path):
    os.mkdir(save_path)

for name in os.listdir(input_path):
    if name.startswith('.'): continue
    input_img = nib.load(os.path.join(input_path, name))

    # input_img = nib.load('/Users/px/Downloads/transfer_64095_files_026b043b (1)/watershed_av.nii.gz')

    input_img = nib.Nifti1Image(np.squeeze(input_img.get_fdata()).astype(np.float32), input_img.affine)

    voxel_before = input_img.header.get_zooms()[0]

    resampled_img = nib.processing.resample_to_output(input_img, voxel_before * 2, order=3)

    nib.save(resampled_img, 'a.nii.gz')

    resampled_img = nib.load('a.nii.gz')

    resampled_img = nib.processing.resample_to_output(resampled_img, voxel_before, order=3)

    nib.save(resampled_img, os.path.join(save_path, name))


input_path = '/Users/px/Downloads/kidney_scans_resampled_1_3'
save_path = '/Users/px/Downloads/kidney_scans_resampled_1_3_cropped_1'

if not os.path.exists(save_path):
    os.mkdir(save_path)


def get_proj(img, axis=0):
    a = np.max(img, axis=axis)
    b = filters.threshold_otsu(a)
    label, num_features = scipy.ndimage.label(a > b)
    largest_cc_mask = label == (np.bincount(label.flat)[1:].argmax() + 1)
    largest_cc_mask = largest_cc_mask.astype(np.uint8)
    from skimage.measure import regionprops

    region_boxes = regionprops(largest_cc_mask)
    minr, minc, maxr, maxc, = region_boxes[0].bbox
    return minr, minc, maxr, maxc



for name in os.listdir(input_path):
    if name.startswith('.'): continue
    img = nib.load(os.path.join(input_path, name))

    affine = img.affine
    img = img.get_fdata()
    m, n, k, *_ = img.shape

    miny, minz, maxy, maxz = get_proj(img, axis=0)
    minx, minz_1, maxx, maxz_1 = get_proj(img, axis=1)
    minx_1, miny_1, maxx_1, maxy_1 = get_proj(img, axis=2)
    minr = min(minx, minx_1)
    minc = min(miny, miny_1)
    min_h = min(minz, minz_1)
    maxr = max(maxx, maxx_1)
    maxc = max(maxy, maxy_1)
    max_h = max(maxz, maxz_1)

    minr = max(0, int(4/5 * minr))
    minc = max(0, int(4/5 * minc))
    min_h = max(0, int(4/5 * min_h))

    maxr = min(m, int(1/5 * (m - maxr) + maxr))
    maxc = min(m, int(1/5 * (n - maxc) + maxc))
    max_h = min(m, int(1/5 * (k - max_h) + max_h))

    cropped = img[minr: maxr, minc: maxc, min_h: max_h]
    ni_img = nib.Nifti1Image(cropped.astype(np.float32), affine)
    nib.save(ni_img, os.path.join(save_path, name))

import matplotlib.patches as mpatches
fig, ax = plt.subplots(figsize=(10, 6))



rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                          fill=False, edgecolor='red', linewidth=2)
ax.add_patch(rect)
ax.set_axis_off()
plt.tight_layout()
plt.show()

plt.imshow(largest_cc_mask, cmap='gray')
plt.show()

def rescale(path_before, path_after):

    img_before = os.path.join(path_before, 'image')
    lab_before = os.path.join(path_before, 'label')

    img_after = os.path.join(path_after, 'image')
    lab_after = os.path.join(path_after, 'label')

    if not os.path.exists(img_after):
        os.mkdir(img_after)

    if not os.path.exists(lab_after):
        os.mkdir(lab_after)

    all_imgs = os.listdir(img_before)[0]

    img_before = os.path.join(img_before, all_imgs)
    lab_before = os.path.join(lab_before, all_imgs)

    img_after = os.path.join(img_after, all_imgs)
    lab_after = os.path.join(lab_after, all_imgs)

    for i, (input_path, output_path) in enumerate(zip([img_before, lab_before], [img_after, lab_after])):
        input_img = nib.load(input_path)
        voxel_before = input_img.header.get_zooms()[0]

        if i == 0:
            resampled_img = nib.processing.resample_to_output(input_img, voxel_before * 3, order=3)
        else:
            resampled_img = nib.processing.resample_to_output(input_img, voxel_before * 3, order=1)

        nib.save(resampled_img, output_path)

rescale('/Users/px/Downloads/transfer_64095_files_026b043b (1)' ,'/Users/px/Downloads/transfer_64095_files_026b043b (1)/down_sampled')

def save_to_disk(img_before, lab_before, img_after, lab_after, voxel_after=0.15):

    for input_path, output_path in zip([img_before, lab_before], [img_after, lab_after]):
        input_img = nib.load(input_path)

        resampled_img = nib.processing.smooth_image(input_img, 0.5)

        # resampled_img = nib.processing.resample_to_output(input_img, voxel_after,
        #                                                   cval=0,order=0,
        #                                                   mode='nearest')

        # data = input_img.get_fdata()
        # data = rescale(data, 0.5,
        #                preserve_range=True).astype(np.float32)
        # affine_func = input_img.affine
        # resampled_img = nib.Nifti1Image(data, affine_func)

        nib.save(resampled_img, output_path)


def blur_save_to_disk(img_before, lab_before, img_after, lab_after, smooth_level=0.5):

    for i, (input_path, output_path) in enumerate(zip([img_before, lab_before], [img_after, lab_after])):
        input_img = nib.load(input_path)

        if i == 0:
            resampled_img = nib.processing.smooth_image(input_img, smooth_level)
        else:
            resampled_img = input_img
        nib.save(resampled_img, output_path)


if __name__ == '__main__':

    path = '/Volumes/T7/Task08_HepaticVessel/Cropped_3/labels/hepaticvessel_001.nii.gz'

    path = '/Volumes/T7/Task08_HepaticVessel/Cropped_3/images/hepaticvessel_001.nii.gz'
    input_img = nib.load(path)
    output_path = '/Users/px/Downloads/a1.nii.gz'

    voxel_before = input_img.header.get_zooms()[0]

    resampled_img = nibabel.processing.resample_to_output(input_img,
                                                          voxel_sizes=(1.0, 1.0, 1.0),
                                                          order = 1
                                                          )




    # resampled_img = nib.processing.smooth_image(input_img, 0.5)
    # resampled_img = nib.processing.resample_from_to(input_img, 1)
    nib.save(resampled_img, output_path)



    dataset_dir = '/Users/px/Downloads/training_same_FOV'

    dataset_dir = '/Users/px/Downloads/jaw_0_115_cropped'

    dataset_dir = '/Users/px/Downloads/jaw_0_115_cropped_upper'

    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    aug_root = os.path.join(dataset_dir, 'aug')
    aug_image_root = os.path.join(aug_root, 'images')
    aug_label_root = os.path.join(aug_root, 'labels')

    safe_make(aug_root)
    safe_make(aug_image_root)
    safe_make(aug_label_root)

    image_names = os.listdir(images_dir)
    label_names = os.listdir(images_dir)
    assert len(image_names) == len(label_names)

    train_15_list = ['Patient_4_0.15mm_cropped.nii.gz', 'Patient_5_0.15mm_cropped.nii.gz',
                     'Patient_15_0.15mm_cropped.nii.gz', 'Patient_25_0.15mm_cropped.nii.gz'
                     ]
    train_15_list = ['Patient_5_0.15mm_cropped.nii.gz',
                     ]
    train_20_list = ['Patient_15_0.15mm_cropped.nii.gz', 'Patient_26_0.15mm_cropped.nii.gz',
                     ]
    train_25_list = ['Patient_25_0.15mm_cropped.nii.gz',
                     ]


    # train_25_list = ['Patient_5_0.15mm_cropped.nii.gz']
    # train_20_list = ['Patient_5_0.15mm_cropped.nii.gz']

    for i in os.listdir(images_dir):
        img_before = os.path.join(os.path.join(images_dir, i))
        lab_before = os.path.join(os.path.join(labels_dir, i))

        # if '15mm' in i or '25mm' in i or '0.2mm' in i:
        if i in train_15_list or i in train_20_list or i in train_25_list:
            voxel_after = 0.3
            voxel_after = 'blurred'
            img_after = os.path.join(os.path.join(aug_image_root, str(voxel_after)+i))
            lab_after = os.path.join(os.path.join(aug_label_root, str(voxel_after)+i))

            # save_to_disk(img_before, lab_before, img_after, lab_after, voxel_after)

            blur_save_to_disk(img_before, lab_before, img_after, lab_after, smooth_level=0.5)

            if i in train_15_list or i in train_20_list:
                voxel_after = 'blurred_7_'
                img_after = os.path.join(os.path.join(aug_image_root, str(voxel_after) + i))
                lab_after = os.path.join(os.path.join(aug_label_root, str(voxel_after) + i))

                blur_save_to_disk(img_before, lab_before, img_after, lab_after, smooth_level=0.7)

            # if '15mm' in i or '0.2mm' in i:
            #     voxel_after = 0.5
            #     img_after = os.path.join(os.path.join(aug_image_root, str(voxel_after)+i))
            #     lab_after = os.path.join(os.path.join(aug_label_root, str(voxel_after)+i))
            #
            #     save_to_disk(img_before, lab_before, img_after, lab_after, voxel_after)