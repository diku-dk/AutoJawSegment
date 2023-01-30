from sklearn.preprocessing import RobustScaler
import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt

path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/hip_val_0930/images/s11-image-cropped.nii'
X = nib.load(path).get_fdata().astype(np.float32)
X = np.reshape(X, (-1, 1))
transformer = RobustScaler().fit(X)
X = transformer.transform(X)
plt.hist(X)
plt.show()

X_01 = (X - X.min())/(X.max() - X.min())
plt.hist(X_01)
plt.show()

plt.hist(np.log(X_01+1e-5))
plt.show()


