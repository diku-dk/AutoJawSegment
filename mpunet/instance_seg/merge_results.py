import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import os
import logging
from mpunet.postprocessing.loggers import *
from mpunet.preprocessing.multi_slice_viewer import multi_show
# log = logging.getLogger(__name__)
# log.debug('My message with %s', 'variable data')

img = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/jawDecAll/images/Patient_25_0.15mm_fixed_dim.nii.gz'
img = nib.load(img).get_fdata()

pred_all = '/Users/px/Downloads/MultiPlanarUNet/jawDecPrimeNormBeginGauss/predictions_0319/nii_files_task_0/Patient_25_0_PRED.nii.gz'
pred_all = nib.load(pred_all).get_fdata()

fg = '/Users/px/Downloads/MultiPlanarUNet/jawDecPrimeNormBeginGauss/predictions_no_argmax/true_foreground5_3_0.8/30Patient_25_0_PRED.nii.gz'
fg = nib.load(fg).get_fdata()

bg = '/Users/px/Downloads/MultiPlanarUNet/jawDecPrimeNormBeginGauss/predictions_no_argmax/true_background/30Patient_25_0_PRED.nii.gz'
bg = nib.load(bg).get_fdata()

prob = '/Users/px/Downloads/Patient_25_0_PRED.nii.gz'
prob = nib.load(prob).get_fdata()
# multi_show(prob, stride=5).matplot3D()

grad = '/Users/px/Downloads/Patient_25_0_PRED_grad.nii.gz'
grad = nib.load(grad).get_fdata()

res = '/Users/px/Downloads/Patient_25_0_PRED 2.nii.gz'
res = nib.load(res).get_fdata()

m, n, k = img.shape
slice = int(k/1.7)

slice = 455

plt.imshow(img[:, slice, :].T, cmap='gray')
plt.show()

jaws = pred_all
jaws = jaws * np.logical_or(jaws == 1, jaws == 4)
jaws[jaws==4] = 2
plt.imshow(jaws[:, slice, :].T, cmap='gray')
plt.show()

plt.imshow(fg[:, 454, :].T,
           cmap='CMRmap'
           )
plt.show()
plt.imshow(bg[:, slice, :].T, cmap='gray')
plt.show()
plt.imshow(res[:, slice, :].T, cmap='CMRmap')
plt.show()
plt.imshow(prob[:, slice, :].T, cmap='gray')
plt.show()

plt.imshow(grad[:, slice, :].T, cmap='gray')
plt.show()

plt.imshow((res+2)[:, slice, :].T,
           cmap='CMRmap'
           )
plt.show()

combined = jaws + (res+2) * (res>0)
combined = (jaws + res.max()) * (jaws>0) + res

plt.imshow(combined[:, slice, :].T,
           cmap='CMRmap'
           )
plt.show()

plt.imshow(img[:, :, slice].T, cmap='gray')
plt.show()

jaws = pred_all[:, :, slice].T
jaws = jaws * np.logical_or(jaws == 1, jaws == 4)
plt.imshow(jaws, cmap='gray')
plt.show()

plt.imshow(fg[:, :, slice].T,
           cmap='CMRmap'
           )
plt.show()
plt.imshow(bg[:, :, slice].T, cmap='gray')
plt.show()
plt.imshow(res[:, :, slice].T, cmap='CMRmap')
plt.show()
plt.imshow(prob[:, 455, :].T, cmap='gray')
plt.show()


plt.imshow(pred_all[:, :, slice].T == 1, cmap='gray')
plt.show()

plt.imshow(res[:, :, slice].T, cmap='CMRmap')
plt.show()

plt.imshow(fg[:, :, slice].T,
           cmap='CMRmap'
           )
plt.show()

plt.imshow(bg[:, :, slice].T, cmap='gray')
plt.show()



lower_root = '/Users/px/Downloads/MultiPlanarUNet/jawDecPrimeNorm/predictions_no_argmax_2/watershed'
upper_root = '/Users/px/Downloads/MultiPlanarUNet/jawDecPrimeNorm/predictions_no_argmax_3/watershed'
predictions_root = '/Users/px/Downloads/MultiPlanarUNet/jawDecPrimeNorm/predictions/nii_files_task_0'
save_root = '/Users/px/Downloads/saves'

file_names = [i for i in os.listdir(predictions_root) if (not i.startswith('.') and (i.endswith('nii') or i.endswith('gz')))]
if not os.path.exists(save_root):
    os.mkdir(save_root)

for name in file_names:
    predictions = os.path.join(predictions_root, name)
    upper_jaw = os.path.join(upper_root, name)
    lower_jaw = os.path.join(lower_root, name)
    save_name = os.path.join(save_root, name)

    predictions = nib.load(predictions)
    affine = predictions.affine
    predictions = predictions.get_fdata().astype(np.uint8)

    upper_jaw = nib.load(upper_jaw).get_fdata().astype(np.uint8)
    lower_jaw = nib.load(lower_jaw).get_fdata().astype(np.uint8)

    lower_bone = predictions == 1
    upper_bone = predictions == 4

    res = np.zeros(predictions.shape, dtype=np.uint8)

    res[lower_bone] = 1
    res[upper_bone] = 2

    global_counter = 3

    n_teeth = len(np.unique(lower_jaw)) - 1
    res += (lower_jaw + 2) * (lower_jaw > 0)

    res += (upper_jaw + 2 + n_teeth) * (upper_jaw > 0)

    res = res * np.logical_not(np.logical_and((upper_jaw > 0), (lower_jaw > 0)))

    a = np.logical_not(np.logical_and((upper_jaw > 0), (lower_jaw > 0))).sum()
    if a > 0:
        print(name, a)

    # for i in np.unique(lower_jaw):
    #     if i == 0: continue
    #     cur_tooth = lower_jaw == i
    #     res[cur_tooth] = global_counter
    #     global_counter += 1
    #
    # for i in np.unique(upper_jaw):
    #     if i == 0: continue
    #     cur_tooth = upper_jaw == i
    #     res[cur_tooth] = global_counter
    #     global_counter += 1

    nib.save(nib.Nifti1Image(res.astype(np.uint8), affine), save_name)
