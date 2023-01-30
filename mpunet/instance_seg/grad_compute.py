from test_threshold import *
from scipy.ndimage import gaussian_gradient_magnitude

img_path = '/Volumes/T7/predictions_no_argmax/preds'
root_path = os.path.dirname(img_path)

save_root = os.path.join(root_path, 'grad')

if not os.path.exists(save_root):
    os.mkdir(save_root)

for i in os.listdir(img_path):
    if i.startswith('.') or not (i.endswith('nii') or i.endswith('gz')):
        continue

    pred_path = os.path.join(img_path, i)
    img_func = nib.load(pred_path)
    pred_img = img_func.get_fdata().astype(np.float32)
    pred_img = np.squeeze(pred_img)

    affine = img_func.affine

    rad_img = gaussian_gradient_magnitude(pred_img, sigma=2)

    rad_img = rad_img.astype(np.float32)

    ni_img = nib.Nifti1Image(rad_img.astype(np.float32)
                             , affine)

    save_path = os.path.join(save_root, i)

    nib.save(ni_img, save_path)
