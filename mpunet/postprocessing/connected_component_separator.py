from scipy.ndimage import label, generate_binary_structure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import os
import cc3d
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star, binary_opening)
from skimage.morphology import (erosion, dilation, opening, closing, binary_closing,
                                binary_dilation, binary_erosion,
                                area_closing,
                                white_tophat, remove_small_holes)


def ccd_largest_part(img):
    res = np.zeros(img.shape)

    for i in np.unique(img):
        if i == 0: continue

        labels_out, num_labels = label((img == i).astype('uint8'))

        lab_list, lab_count = np.unique(labels_out, return_counts=True)

        if lab_list[0] == 0:
            lab_list = lab_list[1:]
            lab_count = lab_count[1:]

        largest_ind = np.argmax(lab_count)
        lab = lab_list[largest_ind]

        res += (i * (labels_out == lab)).astype(np.uint8)

    res = res.astype('uint8')

    return res



def connected_component_separate_3D(file_path, save_path=None, connectivity=26, portion_foreground=0.01,
                                    bg_val=0, n_class=5, erosion=False, show_classes=False):
    img_func = nib.load(file_path)
    affine = img_func.affine
    img = img_func.get_fdata().astype(np.uint8)

    img = np.squeeze(img)
    res = np.empty(img.shape)

    if erosion:
        footprint = ball(3)
        img = binary_erosion(img, footprint)

    cur_label = 1

    for i in np.unique(img):
        if i == 0: continue

        # labels_out = cc3d.connected_components(img, connectivity=connectivity,
        #                                        # max_labels=n_class
        #                                        )  # 26-connected

        from scipy.ndimage import label

        labels_out, num_labels = label((img == i).astype('uint8'))

        avg_volume = np.sum(labels_out != 0) / n_class

        for j in np.unique(labels_out):
            if j == 0:
                continue

            num_voxels = np.sum(labels_out == j)

            if num_voxels > avg_volume * portion_foreground:
                res += (cur_label * (labels_out == j)).astype(np.uint8)
                cur_label += 1
            else:
                print(f"omitting label {i} on {save_path.split('/')[-1]}")

        res = res.astype('uint8')

    # for i, j in enumerate(np.unique(res)):
    #     res[res == j] = i
    #
    # res = res.astype('uint8')

    ni_img = nib.Nifti1Image(res.astype(np.uint8)
                             , affine)
    if show_classes:
        n = len(np.unique(res))
        s = save_path.split('/')
        s[-1] = str(n) + s[-1]
        save_path = '/'.join(s)

    nib.save(ni_img, save_path)


if __name__ == '__main__':

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_weight_map/predictions_0902/nii_files'

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_weight_map_single/predictions/nii_files'

    root_path = '../../my_hip_project_weight_map_origin/predictions_w_fusion/labels_binary'

    root_path = '../../my_hip_project_weight_map_comb_loss_copy/predictions_on_ours/labels_binary'

    file_path = '/Users/px/Downloads/predictions_no_arg/predictions_no_arg/nii_files/Patient_10_0_PRED.nii.gz'
    save_path = '/Users/px/Downloads/predictions_no_arg/predictions_no_arg/nii_files/'
    img_func = nib.load(file_path)
    affine = img_func.affine
    img = img_func.get_fdata().astype(np.float32)

    for i in range(img.shape[-1]):
        save_path_cur = os.path.join(save_path, f'Patient_101_PRED_{i}.nii.gz')
        ni_img = nib.Nifti1Image(img[..., i].astype(np.float32)
                                 , affine)

        nib.save(ni_img, save_path_cur)

    ccd_path = os.path.join(root_path, 'ccd_binary')

    if not os.path.exists(ccd_path):
        os.mkdir(ccd_path)

    for i in os.listdir(root_path):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue

        img_path = os.path.join(root_path, i)

        save_path = os.path.join(ccd_path, i)

        connected_component_separate_3D(img_path, save_path=save_path,
                                        connectivity=6, portion_foreground=0.5,
                                        n_class=5)

#
# image = Image.open(image_path)
# image = np.asarray(image.convert('RGB'))
