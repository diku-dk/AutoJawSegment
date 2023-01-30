import numpy as np
import os

import pandas as pd
import matplotlib.pyplot as plt
import os
import nibabel as nib

from mpunet.preprocessing.multi_slice_viewer import multi_show

def plot_views(views, out_path):
    from mpl_toolkits.mplot3d import Axes3D

    # Create figure, 3D
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Set axes
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(-0.6, 0.6)
    ax.set_zlim(-0.6, 0.6)

    # Plot unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    z_f = np.zeros_like(y)

    ax.plot_surface(x, y, z, alpha=0.1, color="darkgray")
    ax.plot_surface(x, y, z_f, alpha=0.1, color="black")

    # Plot basis axes
    ax.plot([-1, 1], [0, 0], [0, 0], color="blue", linewidth=0.7)
    ax.plot([0, 0], [-1, 1], [0, 0], color="red", linewidth=0.7)
    ax.plot([0, 0], [0, 0], [-1, 1], color="green", linewidth=0.7)

    # Plot views
    for v in views:
        c = np.random.rand(3, )
        ax.scatter(*v, s=50, color=c)
        ax.scatter(*v, s=50, color=c)
        ax.plot([0, v[0]], [0, v[1]], [0, v[2]], color=c, linewidth=2)

        # Plot dashed line to XY plane
        ax.plot([v[0], v[0]], [v[1], v[1]], [0, v[2]], color="gray",
                linewidth=1, linestyle="--")

    ax.view_init(30, -45)
    ax.grid(False)
    ax.axis("off")
    fig.savefig(out_path)


def test_x_y_z():
    img = nib.load('/Users/px/GoogleDrive/MultiPlanarUNet/data_folder/hip_data_0823'
                   '/train/images/s3.nii')
    img = img.get_fdata()
    img = np.squeeze(img)

    m, n, h,  = img.shape

    a = multi_show(img, show_dim=1)
    a.multi_slice_viewer()
    # plt.imshow(img[326, :, :], cmap='gray')

if __name__ == '__main__':

    # test_x_y_z()

    base_path = '/Users/px/GoogleDrive/MultiPlanarUNet/my_hip_project0823_w_val_learn_fusion'

    # views = np.array([0, 1, 0])
    views = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])

    if len(views.shape) < 2:
        views = np.expand_dims(views, axis=0)

    np.savez(os.path.join(base_path, "views_only_3.npz"), views)

    # Plot views
    plot_views(views, os.path.join(base_path, "views_only_3.png"))

    views = np.load(os.path.join(base_path, "views_only_3.npz"))["arr_0"]


    # views = np.load('/Users/px/GoogleDrive/MultiPlanarUNet/mpunet/preprocessing/views.npz')["arr_0"]