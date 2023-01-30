from scipy.ndimage import label, generate_binary_structure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import os

from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_opening)
from skimage.morphology import (erosion, dilation, opening, closing, binary_closing,
                                binary_dilation, binary_erosion,
                                area_closing,
                                white_tophat, remove_small_holes)


def binary_opening_label(img, radius=1, bg_val=0):
    res = np.zeros(img.shape, dtype=np.uint8)

    footprint = ball(radius)

    for label in sorted(np.unique(img)):
        if label == bg_val:
            continue

        img_cur = (img == label).astype(np.uint8)

        labels_out = binary_opening(img_cur, footprint)

        # labels_out = binary_erosion(labels_out, footprint)

        res += (int(label) * labels_out).astype(np.uint8)

    res = res.astype('uint8')

    return res

def binary_erosion_label(img, radius=1, bg_val=0):
    res = np.zeros(img.shape, dtype=np.uint8)

    footprint = ball(radius)

    for label in sorted(np.unique(img)):
        if label == bg_val:
            continue

        img_cur = (img == label).astype(np.uint8)

        labels_out = binary_erosion(img_cur, footprint)

        # labels_out = binary_erosion(labels_out, footprint)

        res += (int(label) * labels_out).astype(np.uint8)

    res = res.astype('uint8')

    return res

def binary_dilation_label(img, radius=1, bg_val=0):
    res = np.zeros(img.shape, dtype=np.uint8)

    footprint = ball(radius)

    for label in sorted(np.unique(img)):
        if label == bg_val:
            continue

        img_cur = (img == label).astype(np.uint8)

        labels_out = binary_dilation(img_cur, footprint)

        # labels_out = binary_erosion(labels_out, footprint)

        res += (int(label) * labels_out).astype(np.uint8)

    res = res.astype('uint8')

    return res


def connected_component_separate_3D(img, connectivity=26, portion_foreground=0.01,
                                    bg_val=0, n_class=5, ):

    img = (img > 0).astype('uint8')

    img = np.squeeze(img)

    res = np.empty(img.shape)

    from scipy.ndimage import label

    labels_out, num_labels = label(img)

    avg_volume = np.sum(labels_out != 0) / 5

    for i in np.unique(labels_out):
        if i == 0:
            continue

        num_voxels = np.sum(labels_out == i)

        if num_voxels > avg_volume * portion_foreground:
            res += i * (labels_out == i)
        else:
            print(f'omitting label {i}')

    res = res.astype('uint8')

    for i, j in enumerate(np.unique(res)):
        res[res == j] = i

    return res.astype('uint8')



if __name__ == '__main__':

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_continue_f_external2/predictions_0902/nii_files/ccd'

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_weight_map_single/predictions/nii_files'

    morph_path = os.path.join(root_path, 'open')

    if not os.path.exists(morph_path):
        os.mkdir(morph_path)

    all_ready_done = os.listdir(morph_path)

    for i in os.listdir(root_path):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue

        img_path = os.path.join(root_path, i)
        img_name = img_path.split('/')[-1]
        save_path = os.path.join(morph_path, img_name)

        if img_name in all_ready_done:
            print(f'skipping image {img_name}')
            continue

        # img_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_weight_map/predictions_0902/nii_files/ccd/s24-image-cropped_PRED.nii.gz'
        # img_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_continue_f_external2/predictions_on_test_ccd_radius_50/nii_files/morph_a70_image_cropped_PRED.nii.gz'

        img = nib.load(img_path)
        data = img.get_fdata()
        affine_func = img.affine

        # res = binary_opening_label(data, radius=2)

        res = binary_erosion_label(data, radius=1)
        res = connected_component_separate_3D(res)
        res = binary_dilation_label(res, radius=1)


        ni_img = nib.Nifti1Image(res.astype(np.uint8)
                                 , affine_func)

        nib.save(ni_img, save_path)

#
# from skimage import morphology
# a = np.array([[0, 0, 0, 0, 0, 0],
#               [0, 1, 0, 1, 1, 0],
#               [0, 1, 0, 1, 1, 0],
#               [0, 0, 0, 0, 0, 0]], bool)
#
# plt.imshow(a)
#
# b = morphology.remove_small_holes(a, area_threshold=12, connectivity=1)
#
# footprint = disk(1)
# b = closing(a, footprint)
#
# plt.imshow(b, cmap='gray')
#
#
# closing(a, disk(1))