from scipy.ndimage import label, generate_binary_structure
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import nibabel.processing as nib_process
import os
import cc3d

def connected_component_legend():
    a = np.array([[0,0,1,1,0,0],
                  [0,0,0,1,0,0],
                  [1,1,0,0,1,0],
                  [0,0,0,1,0,0]])

    str_3D = np.array([[[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]],

                       [[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]],

                       [[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]]], dtype='uint8')

    labeled_array, num_features = label(a)
    structure = np.ones((3,3,3))
    # plt.imshow(labeled_array)

def connected_component_3D(img, connectivity=26, portion_foreground=0.01, bg_val=0):
    
    img = img.astype(np.uint8)

    out = np.zeros(img.shape, dtype=np.uint8)

    for label_cur in np.unique(img):
        if label_cur == bg_val:
            continue

        img_cur = img == label_cur
        img_cur = img_cur.astype(np.uint8)

        # labeled_array, num_features = label(data, structure=str_3D)

        labels_out = cc3d.connected_components(img_cur, connectivity=connectivity) # 26-connected

        num_total_voxels = np.sum(img_cur != 0)

        # var = [np.sum(labels_out == i) > num_total_voxels * portion_foreground
        #        for i in np.unique(labels_out)]

        p = portion_foreground if type(portion_foreground) is not dict \
            else portion_foreground[label_cur]

        for i in np.unique(labels_out):
            if i == 0:
                continue
            num_voxels = np.sum(labels_out == i)
            if i < 10 or i % 10 == 0:
                print(f'num for label {label_cur} part {i} = {num_voxels}')


            if num_voxels > num_total_voxels * p:
                out += label_cur * (labels_out == i)
    
    out = out.astype('uint8')
    
    return out



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

if __name__ == '__main__':

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_weight_map/predictions_on_test_ccd_radius_50_0.1/nii_files'

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_weight_map/predictions_on_external/nii_files'

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project_continue_f_external2/predictions_0902/nii_files'

    root_path = '../../my_hip_project_weight_map_origin/predictions_w_fusion/nii_files'

    root_path = '/Users/px/Downloads/predictions_0914/nii_files'

    root_path = '/Users/px/Downloads/final_kidney_Sep'

    root = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/training_combined/images'

    # for i in os.listdir(root):
    #     if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
    #         continue
    #     img_path = os.path.join(root, i)
    #
    #     img = nib.load(img_path)
    #
    #     shapes = img.shape
    #     print(np.average(shapes[0:3]))

    ccd_path = os.path.join(root_path, 'ccd')

    if not os.path.exists(ccd_path):
        os.mkdir(ccd_path)

    for i in os.listdir(root_path):
        if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
            continue
        img_path = os.path.join(root_path, i)

        img = nib.load(img_path)
        data = img.get_fdata()
        affine_func = img.affine

        res = ccd_largest_part(data)

        save_name = os.path.join(ccd_path, img_path.split('/')[-1])

        ni_img = nib.Nifti1Image(res.astype(np.uint8)
                                 , affine_func)

        nib.save(ni_img, save_name)



#
# image = Image.open(image_path)
# image = np.asarray(image.convert('RGB'))


