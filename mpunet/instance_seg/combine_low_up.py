import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import nibabel as nib
import os

lower_root = '../../jawDecPrimeNorm/predictions_no_argmax_2/images'
upper_root = '../../jawDecPrimeNorm/predictions_no_argmax_3/images'

save_root = '../../jawDecPrimeNorm/predictions_up_low_combined'

file_names = [i for i in os.listdir(lower_root) if (not i.startswith('.') and (i.endswith('nii') or i.endswith('gz')))]
if not os.path.exists(save_root):
    os.mkdir(save_root)

for name in file_names:
    upper_jaw = os.path.join(upper_root, name)
    lower_jaw = os.path.join(lower_root, name)
    save_name = os.path.join(save_root, name)
    affine = nib.load(upper_jaw).affine
    upper_jaw = nib.load(upper_jaw).get_fdata()
    lower_jaw = nib.load(lower_jaw).get_fdata()

    res = nib.Nifti1Image((upper_jaw+lower_jaw).astype(np.float32), affine)

    nib.save(res, save_name)


