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

from connected_component_3D import connected_component_3D

def binary_closing_label(img, radius=1,  bg_val=0):

    res = np.zeros(img.shape, dtype=np.uint8)

    footprint = ball(radius)

    for label in sorted(np.unique(img)):
        if label == bg_val:
            continue

        img_cur = (img == label).astype(np.uint8)

        labels_out = binary_closing(img_cur, footprint)

        # labels_out = binary_erosion(labels_out, footprint)

        res += (int(label) * labels_out).astype(np.uint8)

        res[res > int(label)] = 0

        print(np.unique(res))

    res[res > np.max(img)] = 0

    res = res.astype('uint8')

    return res

def binary_opening_label(img, radius=1,  bg_val=0):

    res = np.zeros(img.shape, dtype=np.uint8)

    footprint = ball(radius)

    for label in sorted(np.unique(img)):
        if label == bg_val:
            continue

        img_cur = (img == label).astype(np.uint8)

        labels_out = binary_opening(img_cur, footprint)

        # labels_out = binary_erosion(labels_out, footprint)

        res += (int(label) * labels_out).astype(np.uint8)

        res[res > int(label)] = 0

        print(np.unique(res))

    res[res > np.max(img)] = 0

    res = res.astype('uint8')

    return res


def binary_erode_label(img, radius=1,  bg_val=0):

    res = np.zeros(img.shape, dtype=np.uint8)

    footprint = ball(radius)

    for label in sorted(np.unique(img)):
        if label == bg_val:
            continue

        img_cur = (img == label).astype(np.uint8)

        labels_out = binary_erosion(img_cur, footprint)

        # labels_out = binary_erosion(labels_out, footprint)

        res += (int(label) * labels_out).astype(np.uint8)

        res[res > int(label)] = 0

        print(np.unique(res))

    res[res > np.max(img)] = 0

    res = res.astype('uint8')

    return res


if __name__ == '__main__':

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_continue_f_external2/predictions_0902/nii_files/ccd'

    root_path = '/Users/px/Downloads/predictions_double/nii_files_task_0/pred_0101'

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/vessels/zoom_in/morph'

    # root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jawAlllabels/eroded'
    morph_path = os.path.join(os.path.dirname(root_path), 'cdd')

    if not os.path.exists(morph_path):
        os.mkdir(morph_path)

    all_ready_done = os.listdir(morph_path)

    for i in reversed(sorted(os.listdir(root_path))):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue
    
        img_path = os.path.join(root_path, i)
        img_name = img_path.split('/')[-1]
        save_path = os.path.join(morph_path, img_name)

        # if img_name in all_ready_done:
        #     print(f'skipping image {img_name}')
        #     continue

        img = nib.load(img_path)
        data = img.get_fdata()
        affine_func = img.affine

        # data = binary_closing_label(data, radius=5)

        # res = binary_opening_label(data, radius=3)
        # res = binary_erode_label(data, radius=2)

        # res = closing(img, ball(5))
        # res = remove_small_holes(data>0, area_threshold=50, connectivity=1)

        data = connected_component_3D(data, connectivity=26, portion_foreground=0.5)


        ni_img = nib.Nifti1Image(data.astype(np.uint8)
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