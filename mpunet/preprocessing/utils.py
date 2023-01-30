import nibabel as nib
import os
import numpy as np
import tensorflow as tf

def loadNii(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    return data


def normImage(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def normImage_tf(img):
    return (img - tf.reduce_min(img)) / (tf.reduce_max(img)
                                              - tf.reduce_min(img))


def reverse_preds(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    affine_func = img.affine
    data = 1 - data
    ni_img = nib.Nifti1Image(data, affine_func)

    save_name = os.path.join(filename)

    nib.save(ni_img, save_name)

