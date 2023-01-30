from scipy.ndimage import label
import numpy as np
import nibabel as nib
import os

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


    pred_root = '/Users/px/Downloads/AllRes/combined_binary'

    pred_path = [i for i in os.listdir(pred_root)
                   if not (i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')))]

    bone_class = 1

    new_label = None
    # new_label = (3, 4)

    for j in sorted(pred_path):
        print(f'working on {j}')
        pred_path = os.path.join(pred_root, j)

        img_func = nib.load(pred_path)
        affine = img_func.affine
        pred = img_func.get_fdata().astype(np.uint8)
        pred = np.squeeze(pred)

        labels_out, num_labels = label(pred == bone_class)
        lab_list, lab_count = np.unique(labels_out, return_counts=True)
        non_largest_ind = np.argsort(lab_count)[:-3]
        labs = lab_list[non_largest_ind]
        pred[np.logical_and(labels_out > 0, np.isin(labels_out, labs))] = 0

        if new_label is not None:
            pred[labels_out == labs[0]] = new_label[0]
            pred[labels_out == labs[1]] = new_label[1]

        nib.save(nib.Nifti1Image((pred).astype(np.uint8), affine), pred_path)
