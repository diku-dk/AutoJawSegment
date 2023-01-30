from test_threshold import *
from scipy.ndimage import distance_transform_edt

def get_distance_minus(y, thresh=1, sum_thresh=20):

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
        sums = d1 + d2
        # w = np.minimum(w, 20)
        w = np.logical_and((w < thresh), sums < sum_thresh)
        w = w.astype(np.uint8)

    else:
        w = np.zeros_like(y)

    return w

def thresholding():

    true_fore_root = '/Volumes/T7/predictions_no_argmax/0.6/12/0.6radius_3'
    root = os.path.dirname(true_fore_root)
    true_back_root = os.path.join(root, 'true_background')

    thresh_list = [0.3, 0.4, 0.5]

    for i in os.listdir(root):
        if not (i.endswith('nii') or i.endswith('gz')): continue

        img = nib.load(os.path.join(root, i))

        img_data = img.get_fdata()

        for thresh in thresh_list:

            res = img_data > thresh
            res = res.astype(np.uint8)

            save_root = os.path.join(root, 'back_'+str(thresh))
            if not os.path.exists(save_root):
                os.makedirs(save_root)
            save_path = os.path.join(save_root, i)
            nib.save(nib.Nifti1Image(res, img.affine), save_path)



if __name__ == '__main__':

    root = '/Users/px/Downloads/predictions_no_argmax'
    true_fore_root = '/Volumes/T7/predictions_no_argmax/0.6/12/0.6radius_3'

    true_back_root = os.path.join(os.path.dirname(true_fore_root), 'true_background')

    dist_thresh = 1
    prob_thresh = 0.5


    for i in os.listdir(true_fore_root):
        if not (i.endswith('nii') or i.endswith('gz')): continue

        img = nib.load(os.path.join(true_fore_root, i))

        img_data = img.get_fdata()

        img_data = img_data.astype(np.uint8)

        res = get_distance_minus(img_data, thresh=dist_thresh)

        save_root = true_back_root
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, i)
        nib.save(nib.Nifti1Image(res, img.affine), save_path)
