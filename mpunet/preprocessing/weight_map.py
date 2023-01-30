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

# root = '/Users/px/Downloads/validation_labels'
#
# files = os.listdir(root)
#
# for i in files:
#     if not i.endswith('nii'): continue
#     i = os.path.join(root, i)
#     name_after = i + '.gz'
#     img = nib.load(i)
#     nib.save(img, name_after)

#
# a = nib.load('/Users/px/Downloads/weight_map_distances/Patient_15_0.15mm_fixed_dim.nii.gz')
#
# res = nib.Nifti1Image((a.get_fdata() < 10).astype(np.uint8), a.affine)
# nib.save(res, ('/Users/px/Downloads/weight_map_distances/1.nii.gz'))


def generate_random_circles(n=100, d=256):
    circles = np.random.randint(0, d, (n, 3))
    x = np.zeros((d, d), dtype=int)
    f = lambda x, y: ((x - x0) ** 2 + (y - y0) ** 2) <= (r / d * 10) ** 2
    for x0, y0, r in circles:
        x += np.fromfunction(f, x.shape)
    x = np.clip(x, 0, 1)

    return x


def unet_weight_map2D(y, wc=None, w0=10, sigma=5):
    # labels = label(y)
    labels = y
    no_labels = labels == 0
    label_ids = sorted(np.unique(labels))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], len(label_ids)), dtype=np.float32)

        for i, label_id in enumerate(label_ids):
            distances[:, :, i] = distance_transform_edt(labels != label_id)

        distances = np.sort(distances, axis=-1)
        d1 = distances[..., 0]
        d2 = distances[..., 1]
        w = w0 * np.exp(-1 / 2 * ((d1 + d2) / sigma) ** 2) * no_labels
    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


def unet_weight_map3D(y, wc=None, w0=10, sigma=5):
    no_labels = y == 0
    label_ids = sorted(np.unique(y))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], y.shape[2], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[..., i] = distance_transform_edt(y != label_id).astype(np.float32)

        distances = distances.astype(np.float32)

        distances = np.sort(distances, axis=-1)
        d1 = distances[..., 0]
        d2 = distances[..., 1]

        del distances

        w = w0 * np.exp(-1 / 2 * ((d1 + d2) ** 2 + (d1 - d2) ** 2 * 0
                                  ) /
                        sigma ** 2) * no_labels

        w = w.astype(np.float32)

        w[w < 0.05] = 0

    else:
        w = np.zeros_like(y)
    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w



def get_distance(y, **kwargs):
    no_labels = y == 0
    label_ids = sorted(np.unique(y))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], y.shape[2], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[..., i] = distance_transform_edt(y != label_id).astype(np.float32)

        distances = distances.astype(np.float32)

        distances = np.sort(distances, axis=-1)
        d1 = distances[..., 0]
        d2 = distances[..., 1]

        del distances

        w = d1 + d2
        w = np.minimum(w, 20)
        w = w.astype(np.float32)

    else:
        w = np.zeros_like(y)

    return w


def get_distance_minus(y, **kwargs):

    label_ids = sorted(np.unique(y))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], y.shape[2], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[..., i] = distance_transform_edt(y != label_id).astype(np.float32)

        distances = distances.astype(np.float32)

        distances = np.sort(distances, axis=-1)
        d1 = distances[..., 0]
        d2 = distances[..., 1]

        del distances

        w = np.abs(d1 - d2)
        w = np.minimum(w, 20)
        w = w.astype(np.float32)

    else:
        w = np.zeros_like(y)

    return w



def get_distance_all(y, **kwargs):
    no_labels = y == 0
    label_ids = sorted(np.unique(y))[1:]

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], y.shape[2], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[..., i] = distance_transform_edt(y != label_id).astype(np.float32)

        distances = distances.astype(np.float32)

        distances = np.sort(distances, axis=-1)
        d1 = distances[..., 0]
        d2 = distances[..., 1]

        del distances

        w1 = d1 + d2
        w1 = np.minimum(w1, 20)
        w1 = w1.astype(np.float32)

        w2 = np.abs(d1 - d2)
        w2 = np.minimum(w2, 20)
        w2 = w2.astype(np.float32)

    return w1.astype(np.float32), w2.astype(np.float32)


def unet_weight_map3D_same_class(y, wc=None, w0=10, sigma=5):
    y, num_labels = label(y)
    no_labels = y == 0

    label_ids = sorted(np.unique(y))[1:]

    print(f'label_ids= {len(label_ids)}')

    if len(label_ids) > 1:
        distances = np.zeros((y.shape[0], y.shape[1], y.shape[2], len(label_ids)))

        for i, label_id in enumerate(label_ids):
            distances[..., i] = distance_transform_edt(y != label_id)

        distances = np.sort(distances, axis=-1)
        d1 = distances[..., 0]
        d2 = distances[..., 1]

        del distances

        w = w0 * np.exp(-1 / 2 * ((d1 + d2) ** 2 + (d1 - d2) ** 2 * 0
                                  ) /
                        sigma ** 2) * no_labels


    else:
        w = np.zeros_like(y)

    if wc:
        class_weights = np.zeros_like(y)
        for k, v in wc.items():
            class_weights[y == k] = v
        w = w + class_weights
    return w


def combine_weights():
    root_path_1 = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecPrime/train/weight_map_morph_teeth_sig1'
    root_path_2 = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecPrime/train/weight_map_no_morph_only_teeth_sig3'
    save_path = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecPrime/train/final_weightmap'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in os.listdir(root_path_1):
        if i.startswith('.'): continue
        path = os.path.join(root_path_1, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img_data = img_func.get_fdata()
        img_data = np.squeeze(img_data)

        path2 = os.path.join(root_path_2, i)
        img_func = nib.load(path2)
        img_data2 = img_func.get_fdata()
        img_data2 = np.squeeze(img_data2)
        img_data = img_data + np.sqrt(10) * np.sqrt(img_data2)

        img_data = img_data.astype(np.float32)
        ni_img = nib.Nifti1Image(img_data, affine)
        nib.save(ni_img, os.path.join(save_path, i))


def save_only_teeth(images_root):

    images_root = '/Users/px/Downloads/labels_indiv_teeth_5'

    label_root = os.path.join(images_root, 'labels_indiv_teeth')
    weight_root = os.path.join(images_root, 'labels_only_tooth')

    if not os.path.exists(weight_root):
        os.mkdir(weight_root)

    for i in os.listdir(label_root):
        if i.startswith('.'): continue

        path = os.path.join(label_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img_data = img_func.get_fdata()
        img_data = np.squeeze(img_data)

        vals = np.sort(np.unique(img_data))[1:]
        areas = np.array([np.sum(img_data == i) for i in vals])
        ranks = np.argsort(areas)
        img_data[img_data == vals[ranks[-1]]] = 0
        img_data[img_data == vals[ranks[-2]]] = 0

        ni_img = nib.Nifti1Image(img_data.astype(np.uint8), affine)
        save_path = os.path.join(weight_root, i)
        nib.save(ni_img, save_path)


def get_weight_from_distance(y, w0=20, min_val=1, max_val=10,  wc=False):

    y = np.maximum(- w0 * np.log((y - min_val + 1e-10)/max_val), 0)
    y = y.astype(np.float32)
    y[y < 0.1] = 0
    y = y.astype(np.float32)

    return y


def get_gauss_from_distance(y, w0=20, sigma=3):

    w = w0/sigma * np.exp(-1 / 2 * (y/sigma) ** 2)
    w = w.astype(np.float32)
    w[w < 0.1] = 0
    w = w.astype(np.float32)

    return w

def filter_out_small(distance_root):
    distance_root = '/Users/px/Downloads/MultiPlanarUNet/data_folder/training_combined/weight_map'

    for i in os.listdir(distance_root):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('nii.gz')): continue

        path = os.path.join(distance_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img = img_func.get_fdata()
        img = np.squeeze(img)
        img[img < 0.01] = 0
        img = img.astype(np.float32)

        nib.save(nib.Nifti1Image(img, affine), path)


def threshold_minus(images_root):

    distance_root = os.path.join(images_root, 'weight_map_distances')
    distance_minus_root = os.path.join(images_root, 'weight_map_distances_minus')

    distance_minus_threshold_root = os.path.join(images_root, 'weight_map_distances_minus_barrier')

    if not os.path.exists(distance_minus_threshold_root):
        os.mkdir(distance_minus_threshold_root)

    for i in os.listdir(distance_root):
        if i.startswith('.'): continue

        path = os.path.join(distance_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        distance = img_func.get_fdata()
        distance = np.squeeze(distance)
        distance = distance.astype(np.float32)

        path = os.path.join(distance_minus_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        distance_minus = img_func.get_fdata()
        distance_minus = np.squeeze(distance_minus)
        distance_minus = distance_minus.astype(np.float32)

        save_path = os.path.join(distance_minus_threshold_root, i)

        # res = get_gauss_from_distance(distance_minus, w0=40, sigma=4) * (distance < 10)
        res = get_weight_from_distance(distance_minus, min_val=0, max_val=10) * (distance < 10)

        res = res.astype(np.float32)

        nib.save(nib.Nifti1Image(res, affine), save_path)

        del res, distance, distance_minus


if __name__ == '__main__':


    images_root = '/Users/px/Downloads/MultiPlanarUNet/data_folder/hips/train'


    # label_root = os.path.join(images_root, 'labels_indiv_teeth')

    label_root = os.path.join(images_root, 'labels')


    distance_root = os.path.join(images_root, 'weight_map_distances')
    # label_root = os.path.join(images_root, '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jawDecPrime/labels_indiv_teeth')

    # label_root = os.path.join(images_root, 'weight_map_distances_minus_threshold')

    distance_minus_root = os.path.join(images_root, 'weight_map_distances_minus')

    # weight_root = os.path.join(images_root, 'weight_map_morph_teeth')

    weight_root = os.path.join(images_root, 'weight_maps_new')

    morph = True

    if not os.path.exists(weight_root):
        os.mkdir(weight_root)

    if not os.path.exists(distance_root):
        os.mkdir(distance_root)

    if not os.path.exists(distance_minus_root):
        os.mkdir(distance_minus_root)

    for i in os.listdir(label_root):
        if i.startswith('.'): continue

        path = os.path.join(label_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img_data = img_func.get_fdata()

        img_data = np.squeeze(img_data)

        if morph:
            img_data = binary_erosion_label(img_data, radius=2)

        img_data = img_data.astype(np.float32)
        #
        # weight_map = get_gauss_from_distance(img_data)
        # save_path = os.path.join(weight_root, i)
        # nib.save(nib.Nifti1Image(weight_map, affine), save_path)

        weight_map = unet_weight_map3D(img_data, w0=20 if morph else 20, sigma=5)
        weight_map = weight_map.astype(np.float32)
        ni_img = nib.Nifti1Image(weight_map, affine)
        save_path = os.path.join(weight_root, i)
        nib.save(ni_img, save_path)

        continue


        distance, distance_minus = get_distance_all(img_data)

        nib.save(nib.Nifti1Image(distance, affine), os.path.join(distance_root, i))
        nib.save(nib.Nifti1Image(distance_minus, affine), os.path.join(distance_minus_root, i))


        save_path = os.path.join(weight_root, i)

        res = get_weight_from_distance(distance_minus, min_val=0, max_val=5) * (distance < 5)
        res = res.astype(np.float32)

        nib.save(nib.Nifti1Image(res, affine), save_path)

        # weight_map = get_weight_from_distance(img_data)
        # weight_map = unet_weight_map3D_same_class(img_data==2, w0=20 if morph else 10, sigma=5)


def test_funcs():
    y = generate_random_circles()

    wc = {
        0: 1,  # background
        1: 1  # objects
    }

    w = unet_weight_map2D(y, wc)

    fig, axs = plt.subplots(ncols=2, squeeze=False)

    axs[0, 0].imshow(img_data[110], cmap='gray')
    # axs[0, 0].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    # axs[0, 1].imshow(w, #cmap='gray'
    #                  )
    img = axs[0, 1].imshow(weight_map[110] * (weight_map[110] > 1),  # alpha=lab_alpha,
                           cmap='seismic')
    plt.show()

    fig.colorbar(img)
    fig.tight_layout()

    plt.show()

    import tensorflow as tf

    y_true = np.array([[1, 0], [0, 1]])

    y_true = np.expand_dims(y_true, axis=-1)

    y_pred = np.array([[0.5, 1], [1, 0]])
    y_pred_res = 1 - y_pred
    y_pred = np.stack([y_pred, y_pred_res], axis=-1)

    # Using 'auto'/'sum_over_batch_size' reduction type.
    scce = tf.keras.losses.SparseCategoricalCrossentropy()
    scce(y_true, y_pred).numpy()

    scce(y_true, y_pred, tf.constant([0.3, 0.7])).numpy()

    scce(y_true, y_pred, tf.constant([[1, 2], [2, 2]])).numpy()

# TP = 0
# for i in np.unique(a):
#     a_c = a==i
#     b_c = b==i
#     TP =

    path1 = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecLabelsAll/weight_map_distances_minus_barrier'
    path2 = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecPrime/train/weight_map_distances_minus_threshold_barrier'
    save = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecPrime/train/weight_final_barrier'

    if not os.path.exists(save):
        os.mkdir(save)

    for i in os.listdir(path2):
        if i.startswith('.'): continue
        p1 = os.path.join(path1, i)
        p2 = os.path.join(path2, i)
        p3 = os.path.join(save, i)

        img1 = nib.load(p1)
        img2 = nib.load(p2)

        res = (img1.get_fdata() + img2.get_fdata())/2
        res = res.astype(np.float32)
        res = nib.Nifti1Image(res, img1.affine)
        nib.save(res, p3)