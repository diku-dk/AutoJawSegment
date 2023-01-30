import numpy as np
from mpunet.interpolation import RegularGridInterpolator
from scipy.ndimage.filters import gaussian_filter
import skimage
import matplotlib.pyplot as plt
from skimage.transform import rescale, resize, downscale_local_mean

# from scipy.interpolate import RegularGridInterpolator

image = skimage.data.camera()
image = downscale_local_mean(image, (4, 3))

plt.imshow(image, cmap='gray')
plt.show()

shape = image.shape[:2]
channels = image.shape[-1]
dtype = image.dtype
bg_val = 0

# Define coordinate system
coords = np.arange(shape[0]), np.arange(shape[1])

# Initialize interpolators
im_intrps = RegularGridInterpolator(coords, image,
                                     method="linear",
                                     bounds_error=False,
                                     fill_value=bg_val,
                                     dtype=np.float32)

sigma = 20
alpha = 200
# Get random elastic deformations
dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                     mode="constant", cval=0.) * alpha
dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma,
                     mode="constant", cval=0.) * alpha

# Define sample points
x, y = np.mgrid[0:shape[0], 0:shape[1]]
indices = np.reshape(x + dx, (-1, 1)), \
          np.reshape(y + dy, (-1, 1))

# Interpolate all image channels
image = np.empty(shape=image.shape, dtype=dtype)

a = im_intrps(np.squeeze(np.array(indices)).T).reshape(shape)
b = im_intrps((indices)).reshape(shape)

coords = np.linspace(0, shape[0], 128), np.linspace(0, shape[1], 128)
c_x, c_y = shape[0]//2, shape[1]//2
# coords = np.linspace(-64, 64, 128) + c_x, np.linspace(-64, 64, 128) + c_y

grid = np.meshgrid(coords[0], coords[1], indexing='ij')

x, y = grid
indices = np.reshape(x, (-1, 1)), \
          np.reshape(y, (-1, 1))

b = im_intrps((indices)).reshape((128, 128))
plt.imshow(b, cmap='gray')
plt.show()

