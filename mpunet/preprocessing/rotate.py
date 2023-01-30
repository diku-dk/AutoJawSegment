import nibabel as nib
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage
from scipy.spatial.transform import Rotation as R
# from mpunet.interpolation import RegularGridInterpolator
from scipy.interpolate import RegularGridInterpolator

def affine_rotate():
    cos_gamma = np.cos(np.pi / 4)
    sin_gamma = np.sin(np.pi / 4)
    rotation_affine = np.array([[1, 0, 0, 0],
                                [0, cos_gamma, -sin_gamma, 0],
                                [0, sin_gamma, cos_gamma, 0],
                                [0, 0, 0, 1]])
    affine_so_far = rotation_affine.dot(affine)

    rotation_affine = np.array([[cos_gamma, -sin_gamma, 0, 0],
                                [sin_gamma, cos_gamma, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])

    affine_so_far = rotation_affine.dot(affine_so_far)

def random_rotation_3d(batch, max_angle):
    """ Randomly rotate an image by a random angle (-max_angle, max_angle).

    Arguments:
    max_angle: `float`. The maximum rotation angle.

    Returns:
    batch of rotated 3D images
    """
    r = R.from_rotvec(np.array([np.pi, np.pi, np.pi])/8)
    batch = r.apply(batch)

    return batch

    size = batch.shape

    image1 = np.squeeze(batch)
    # rotate along z-axis

    angle = max_angle

    # angle = np.random.uniform(-max_angle, max_angle)
    image2 = scipy.ndimage.interpolation.rotate(image1, angle, mode='constant', axes=(0, 1), reshape=False)

    # rotate along y-axis
    # angle = np.random.uniform(-max_angle, max_angle)
    image3 = scipy.ndimage.interpolation.rotate(image2, angle, mode='constant', axes=(0, 2), reshape=False)

    # rotate along x-axis
    # angle = np.random.uniform(-max_angle, max_angle)
    batch_rot = scipy.ndimage.interpolation.rotate(image3, angle, mode='constant', axes=(1, 2), reshape=False)
    #                print(i)

    return batch_rot.reshape(size)



if __name__ == '__main__':

    root_path = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecLabelsAll'

    labels_path = os.path.join(root_path, 'labels')

    res_path = os.path.join(root_path, 'labels_rotated_manual_align_center')

    if not os.path.exists(res_path):
        os.mkdir(res_path)


    for i in os.listdir(labels_path):
        if i.startswith('.'):
            continue
        path = os.path.join(labels_path, i)
        save_path = os.path.join(res_path, i)

        img_func = nib.load(path)
        affine = img_func.affine


        img_data = img_func.get_fdata()
        img_data = img_data.astype('uint8')
        shape = np.array(img_data.shape[:3])
        coords = np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2])
        bg_val = 0
        # img_data = np.expand_dims(img_data, axis=-1)
        # img_data = np.repeat(img_data, 2, axis=-1)

        im_intrps = RegularGridInterpolator(coords, img_data,
                                            method="nearest",
                                            bounds_error=False,
                                            fill_value=bg_val,
                                            # dtype=np.uint8
                                            )

        xg, yg, zg = np.meshgrid(coords[0], coords[1], coords[2], indexing='ij')
        points = np.c_[xg.ravel(), yg.ravel(), zg.ravel()]

        del xg, yg, zg

        center = shape//2

        points -= center
        points = random_rotation_3d(points, max_angle=np.pi/3)
        points += center

        img_data = im_intrps((points)).reshape(shape)

        del im_intrps, points

        img_data = img_data.astype('uint8')

        ni_img = nib.Nifti1Image(img_data, affine)
        nib.save(ni_img, save_path)