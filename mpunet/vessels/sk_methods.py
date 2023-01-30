seed = 42  # for reproducibility


import nibabel as nib
import nibabel.processing
from skimage.transform import rescale

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from skimage import color, data, filters, graph, measure, morphology


path = '/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/vessels/train/labels/resacaled_cropped_RAT_03_reshaped.nii.gz'

path = '/Users/px/Downloads/a.nii.gz'
output_path = '/Users/px/Downloads/b.nii.gz'

input_img = nib.load(path)
img = input_img.get_fdata()
affine = input_img.affine

img = img.astype(np.float32)

img = img.astype(np.uint8)


skeleton = morphology.skeletonize_3d(img)
skeleton = morphology.skeletonize(img, method=None)

skel, distance = morphology.medial_axis(img, return_distance=True)
dist_on_skel = distance * skel

retina_source = data.retina()

_, ax = plt.subplots()
ax.imshow(retina_source)
ax.set_axis_off()
_ = ax.set_title('Human retina')
retina = color.rgb2gray(retina_source)
t0, t1 = filters.threshold_multiotsu(retina, classes=3)
mask = (retina > t0)
vessels = filters.sato(retina, sigmas=range(1, 10))

plt.imshow(vessels)
plt.show()

vessels = vessels * mask

_, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(retina, cmap='gray')
axes[0].set_axis_off()
axes[0].set_title('grayscale')
axes[1].imshow(vessels, cmap='magma')
axes[1].set_axis_off()
_ = axes[1].set_title('Sato vesselness')


t0, t1 = filters.threshold_multiotsu(img, classes=3)
mask = (img > t0)

vessels = filters.sato(img, sigmas=range(1, 10), mode='constant')

vessels = vessels * mask



_, axes = plt.subplots(nrows=1, ncols=2)
axes[0].imshow(img, cmap='gray')
axes[0].set_axis_off()
axes[0].set_title('grayscale')
axes[1].imshow(vessels, cmap='magma')
axes[1].set_axis_off()
_ = axes[1].set_title('Sato vesselness')


thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
labeled = ndi.label(thresholded)[0]

_, ax = plt.subplots()
ax.imshow(color.label2rgb(labeled, img))
ax.set_axis_off()
_ = ax.set_title('thresholded vesselness')

largest_nonzero_label = np.argmax(np.bincount(labeled[labeled > 0]))
binary = labeled == largest_nonzero_label
skeleton = morphology.skeletonize(binary)
g, nodes = graph.pixel_graph(skeleton, connectivity=2)
px, distances = graph.central_pixel(
        g, nodes=nodes, shape=skeleton.shape, partition_size=100
        )

centroid = measure.centroid(labeled > 0)



res = nib.Nifti1Image(vessels, affine)
nib.save(res, output_path)

res = nib.Nifti1Image(skeleton, affine)
nib.save(res, output_path)

_, ax = plt.subplots()
ax.imshow(color.label2rgb(skeleton, img))
ax.scatter(px[1], px[0], label='graph center')
ax.scatter(centroid[1], centroid[0], label='centroid')
ax.legend()
ax.set_axis_off()
ax.set_title('vessel graph center vs centroid')

plt.show()


