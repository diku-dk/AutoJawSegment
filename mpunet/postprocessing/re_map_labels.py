import numpy as np
import nibabel as nib
import os
a = np.array([1, 2, 6, 4, 2, 3, 2])
a = np.array([[1, 2, 6, ],
              [2, 3, 2]])

def remapping(a):
    u, indices = np.unique(a, return_inverse=True)
    true_ranges = np.arange(len(u))
    remapped = true_ranges[indices].reshape(a.shape)
    assert (np.all(a == u[indices].reshape(a.shape)))
    return remapped

teeth_root = '/Users/px/Downloads/jawVal/watershed'
all_root = '/Users/px/Downloads/jawVal/orig'

save_root = '/Users/px/Downloads/jawVal/mapped'

file_names = [i for i in os.listdir(teeth_root) if (not i.startswith('.') and (i.endswith('nii') or i.endswith('gz')))]
if not os.path.exists(save_root):
    os.mkdir(save_root)


for name in file_names:
    teeth_path = os.path.join(teeth_root, name)
    all_path = os.path.join(all_root, name)
    save_path = os.path.join(save_root, name)

    img_func = nib.load(teeth_path)
    affine = img_func.affine
    teeth = img_func.get_fdata()

    img_func = nib.load(all_path)
    affine = img_func.affine
    bones = img_func.get_fdata()

    bones[np.logical_or(bones == 2, bones == 3)] = 0

    teeth += 5
    teeth[teeth == 5] = 0

    combined = (teeth == 0) * bones + teeth

    combined = remapping(combined)

    ni_img = nib.Nifti1Image(combined.astype(np.uint8)
                             , affine)

    nib.save(ni_img, save_path)
