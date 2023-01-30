import os
import os
from shutil import copyfile

import numpy as np

import nibabel as nib
import os
import numpy as np

def loadNii(filename):
    img = nib.load(filename)
    data = img.get_fdata()
    return data


def normImage(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def normImage_tf(img):
    import tensorflow as tf
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



def get_teeth_label_number(txt_path, class_name='teeth_mandibular'):

    require_teeth = True if 'teeth' in class_name.lower() else False

    with open(txt_path) as f:
        txt = f.read()
        txt = txt.split('#')[-1]
        txt = txt.split('\n')[1:]
        txt = [i.split(' ')[:2] for i in txt if len(i) > 0]
        for label, label_name in txt:
            has_teeth =True if 'teeth' in label_name.lower() else False
            if class_name.lower() in label_name.lower() and require_teeth == has_teeth:
                teeth_mandibular_label = int(label)
                # print(f'{txt_path}: {teeth_mandibular_label}')
                break
    return teeth_mandibular_label

def get_all_legend(colab=False):
    root_path = '/Users/px/Downloads/jaw_two_patient_June10'
    if colab:
        root_path = '/content/drive/MyDrive/jaw data'

    file_names = os.listdir(root_path)

    for file in file_names:
        file_dir = os.path.join(root_path, file)
        if not os.path.isdir(file_dir):
            continue

        txt_path = os.path.join(file_dir, 'Segmentation-label_ColorTable.txt')
        label_path = os.path.join(file_dir, 'Segmentation-label.nii')
        label_data = loadNii(label_path)
        func = nib.load(label_path)

        teeth_mandibular_label = get_teeth_label_number(txt_path, class_name='teeth_mandibular')
        mandible = get_teeth_label_number(txt_path, class_name='mandible')


        print(f'shape: {label_data.shape}')
        label_teeth = (label_data == teeth_mandibular_label).astype(np.int8)
        label_mandible = (label_data == mandible).astype(np.int8) * 2

        try:
            teeth_maxillary = get_teeth_label_number(txt_path, class_name='teeth_maxillary')
            maxillary = get_teeth_label_number(txt_path, class_name='Maxilla')

            label_teeth_maxillary= (label_data == teeth_maxillary).astype(np.int8) * 3
            label_maxillary = (label_data == maxillary).astype(np.int8) * 4

        except:
            label_teeth_maxillary = 0
            label_maxillary = 0

        final_label = (label_mandible + label_teeth +
                       label_teeth_maxillary + label_maxillary
                       ).astype(np.int8)

        ni_img = nib.Nifti1Image(final_label, func.affine)
        outpath = os.path.join(file_dir, 'jaw_all_labels.nii')
        nib.save(ni_img, outpath)
        

def separate_imgaes_labels(colab=False):
    root_path = '/Users/px/Downloads/jaw_two_patient_June10'
    if colab:
        root_path = '/content/drive/MyDrive/jaw data'

    after_root = os.path.join(os.path.dirname(root_path),
                              'jaw_all_labels')
    print(f'after: {after_root}')
    images_root = os.path.join(after_root, 'images')
    labels_root = os.path.join(after_root, 'labels')

    if not os.path.exists(after_root):
        os.mkdir(after_root)

    if not os.path.exists(images_root):
        os.mkdir(images_root)

    if not os.path.exists(labels_root):
        os.mkdir(labels_root)

    file_names = os.listdir(root_path)

    for file in file_names:
        file_dir = os.path.join(root_path, file)
        if not os.path.isdir(file_dir) or 'images' in file or 'labels' in file:
            continue

        image_path_before = os.path.join(file_dir, 'CBCT_scan.nii')
        if not os.path.exists(image_path_before):
            image_path_before = os.path.join(file_dir, 'CBCT_scan_cropped.nii')

        label_path_before = os.path.join(file_dir, 'jaw_all_labels.nii')

        image_path_after = os.path.join(images_root, file+'.nii')
        label_path_after = os.path.join(labels_root, file+'.nii')

        copyfile(image_path_before, image_path_after)
        copyfile(label_path_before, label_path_after)


def expand_dims_images(colab=False):
    root_path = '/Users/px/Downloads/jaw_all_labels/'
    if colab:
        root_path = '/content/drive/MyDrive/mandible_mandibular_separate/'

    images_root = os.path.join(root_path, 'images')
    labels_root = os.path.join(root_path, 'labels')

    file_names = os.listdir(images_root)

    for file in file_names:
        image_file_dir = os.path.join(images_root, file)
        label_file_dir = os.path.join(labels_root, file)

        img = loadNii(image_file_dir).astype(np.float32)
        img_func = nib.load(image_file_dir)
        img = np.expand_dims(img, axis=-1)
        ni_img = nib.Nifti1Image(img, img_func.affine)
        nib.save(ni_img, image_file_dir)

        label = loadNii(label_file_dir).astype(np.int8)
        label_func = nib.load(label_file_dir)
        label = np.expand_dims(label, axis=-1)
        ni_label = nib.Nifti1Image(label, label_func.affine)
        nib.save(ni_label, label_file_dir)



if __name__ == '__main__':
    get_all_legend(colab=False)
    separate_imgaes_labels(colab=False)
    expand_dims_images(colab=False)

def try_save():
    image_file_dir = '/Users/px/GoogleDrive/MultiPlanarUNet/dataset_hip/dataset0602/train/images/image_aa7m_crop.nii'
    image_file_dir = '/Users/px/Downloads/image_aa7m_crop.nii'
    img = loadNii(image_file_dir).astype(np.float32)
    img_func = nib.load(image_file_dir)
    img = np.expand_dims(img, axis=-1)
    ni_img = nib.Nifti1Image(img, img_func.affine)
    nib.save(ni_img, image_file_dir)

    label_file_dir = '/Users/px/GoogleDrive/MultiPlanarUNet/dataset_hip/dataset0602/train/labels/image_aa7m_crop.nii'

    label = loadNii(label_file_dir).astype(np.int8)
    label_func = nib.load(label_file_dir)
    label = np.expand_dims(label, axis=-1)
    ni_label = nib.Nifti1Image(label, label_func.affine)
    nib.save(ni_label, label_file_dir)

    import nrrd
    nrrd_path = '/Users/px/Downloads/nii_data_folder/hip data/labelmap/labelmap-staircase/labelmap-staircase.nrrd'
    # Read the data back from file
    readdata, header = nrrd.read(nrrd_path)
    print(readdata.shape)
    print(header)

