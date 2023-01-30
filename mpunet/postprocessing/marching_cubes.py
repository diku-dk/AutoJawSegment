import sys, os
import numpy as np

import nibabel as nib
from scipy import ndimage

# from pyqtgraph.opengl import GLViewWidget, MeshData
# from pyqtgraph.opengl.items.GLMeshItem import GLMeshItem

from skimage import measure
import nibabel as nib
# from PyQt5.QtGui import QApplication

import mcubes


def marching_cubes(res, name='vessel', level=0.9, spacing=1, method=None, smooth=False, show=False):
    if smooth:
        res = mcubes.smooth(res)

    if method == 'skimage':
        verts, faces, normals, values = measure.marching_cubes(volume=res,
                                                               # level=level,
                                                               # spacing=spacing
                                                                       )
    else:
        verts, faces = mcubes.marching_cubes(res, 0)

    mcubes.export_obj(verts, faces, name+'.obj')

    # mcubes.export_mesh(verts, faces, name+'.dae', "MySphere")

    if show:
        app = QApplication([])
        view = GLViewWidget()

        mesh = MeshData(verts / 200, faces)  # scale down - because camera is at a fixed position
        # or mesh = MeshData(smooth_vertices / 100, faces)
        mesh._vertexNormals = normals
        # or mesh._vertexNormals = smooth_normals

        item = GLMeshItem(meshdata=mesh, color=[1, 0, 0, 1], shader="normalColor")

        view.addItem(item)
        view.show()
        app.exec_()


if __name__ == '__main__':

    root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/ubuntu_predictions/all/all'
    save_root = '/Users/px/GoogleDrive/MultiPlanarUNet/ubuntu_predictions/all/surface_mesh'

    root_path = '/Users/px/Documents/GitHub/cziAnalysis/data_folder'
    save_root = '/Users/px/Documents/GitHub/cziAnalysis/surface'

    root_path = '/Users/px/Downloads/final_kidney_Sep/whole_structure'
    save_root = '/Users/px/Downloads/final_kidney_Sep/whole_structure_marching'

    if not os.path.exists(save_root):
        os.mkdir(save_root)
    # root_path = '/Users/px/GoogleDrive/MultiPlanarUNet/all_labels_only_decoder/predictions_by_radius/nii_files'


    for i in os.listdir(root_path):
        if i.startswith('.'): continue
        # if i != 'largest_part_veseel_254.nii.gz': continue
        img_path = os.path.join(root_path, i)

        # img_path = '/Users/px/Downloads/seg/1.nii.gz'

        img = nib.load(img_path)
        data = img.get_fdata()
        affine_func = img.affine

        save_name = img_path.split('/')[-1]
        save_name = save_name.split('.')[0]
        save_name = os.path.join(save_root, save_name)


        marching_cubes(data, name=save_name, level=0.9, smooth=True, method='mcube', show=False)
