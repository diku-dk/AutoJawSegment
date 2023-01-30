import nibabel as nib
import numpy as np
import os


def simple_thresholding(file_path, save_path, thresh=0, to_label=False):
    img_func = nib.load(file_path)
    affine = img_func.affine
    img_data = img_func.get_fdata()

    # img_data = img_data[np.where(img_data>0)[0]]
    if to_label:
        img_data = (img_data > thresh).astype('uint8')

    else:
        img_data = img_data * (img_data > thresh)
        img_data = img_data.astype('float32')

        if img_data.shape[-1] != 1:
            img_data = np.expand_dims(img_data, axis=-1)

    ni_img = nib.Nifti1Image(img_data, affine)

    nib.save(ni_img, save_path)


if __name__ == '__main__':

    file_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/hip_data_0823/test/images/s11-image-cropped.nii'

    file_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/our_images_cropped0824/images/a7w_image_cropped.nii'

    file_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/hips/train/weight_map/a7m-image-cropped.nii'

    file_path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/testdata0902/images/s17-image-cropped.nii'


    root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/hip_data_0823/test/images'

    root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/testdata0902/images'

    file_path = '/Users/px/Downloads/test-to-training/images/s13-image-cropped.nii'


    root = '/Users/px/GoogleDrive/MultiPlanarUNet/jaw0920_w_gap/weight_map_morph_1'

    root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/hip_no_used_1004/images'

    for i in os.listdir(root):
        if i.startswith('.'):
            continue
        path = os.path.join(root, i)
        save_path = os.path.join(root, #'seg_'+i
                                  i)
        simple_thresholding(file_path=path, save_path=save_path,
                            thresh=0,
                            to_label=False
                            )
