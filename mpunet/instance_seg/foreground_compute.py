
from test_threshold import *

def morph_from_threshed():
    thresh = 0.6
    radius = 3
    root = '/Users/px/Downloads/predictions_no_argmax/0.6/12'
    for i in os.listdir(root):
        if not (i.endswith('nii') or i.endswith('gz')): continue

        img = nib.load(os.path.join(root, i))

        img_data = img.get_fdata()

        res = img_data > thresh
        res = res.astype(np.uint8)
        res = res.squeeze()
        res = binary_opening(res, ball(radius))

        # res = binary_erosion(res, ball(1))

        res = ccd_separate(res)

        res = res.astype(np.uint8)

        save_root = os.path.join(root, str(thresh)+f'radius_{radius}')
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, i)
        nib.save(nib.Nifti1Image(res, img.affine), save_path)


def ccd_separate(img, portion_foreground=0.05):
    img = np.squeeze(img)
    res = np.empty(img.shape)

    cur_label = 1

    for i in np.unique(img):
        if i == 0: continue

        from scipy.ndimage import label

        labels_out, num_labels = label((img == i).astype('uint8'))

        avg_volume = np.sum(labels_out != 0) / num_labels

        for j in np.unique(labels_out):
            if j == 0:
                continue

            num_voxels = np.sum(labels_out == j)

            if num_voxels > avg_volume * portion_foreground:
                res += (cur_label * (labels_out == j)).astype(np.uint8)
                cur_label += 1
            else:
                print(f"omitting label {i}")

        res = res.astype('uint8')

    return res

if __name__ == '__main__':

    root = '/Users/px/Downloads/predictions_no_argmax'
    save_root = os.path.join(os.path.dirname(root), 'true_foreground')

    if not os.path.exists(save_root):
        os.makedirs(save_root)

    thresh = 0.6
    radius = 3

    for i in os.listdir(root):
        if not (i.endswith('nii') or i.endswith('gz')): continue

        path = os.path.join(root, i)

        # path = '/Volumes/T7/predictions_no_argmax/Patient_12_0_PRED.nii.gz'

        img = nib.load(path)

        img_data = img.get_fdata()


        res = img_data > thresh
        res = res.astype(np.uint8)

        res = binary_opening(res, ball(radius))

        res = ccd_separate(res)
        res = res.astype(np.uint8)

        # save_root = os.path.join(os.path.dirname(root), str(thresh))

        save_path = os.path.join(save_root, i)

        nib.save(nib.Nifti1Image(res, img.affine), save_path)


