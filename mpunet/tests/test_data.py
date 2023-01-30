import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os
#%matplotlib inline
from multi_slice_viewer import *



def aug_test(x,y,alpha = 200,sigma=25):
    from mpunet.augmentation.elastic_deformation import elastic_transform_2d
    augmented_x, augmented_y = [], []
    image, labels = elastic_transform_2d(x, y, alpha, sigma, bg_val = 0.0)
    augmented_x.append(image)
    augmented_y.append(labels)
    return augmented_x,augmented_y
  # augmenters: [
  #   {cls_name: "Elastic2D",
  #    kwargs: {alpha: [0, 450], sigma: [20, 30], apply_prob: 0.333}}
  # ]


if __name__ == '__main__':

    os.chdir('/Users/px/GoogleDrive/MultiPlanarUNet')
    print(os.getcwd())
    print(os.listdir())
    data_path = '/Users/px/GoogleDrive/3DUnetCNN/examples/brats2020/MICCAI_BraTS2020_TrainingData/' \
                'BraTS20_Training_022/BraTS20_Training_022_t1.nii.gz'
    data_path = "/Users/px/Downloads/test_001_V00.nii.gz"
    data_path = './data_folder/train/images/prostate_00.nii.gz'

    mask_path = './data_folder/train/labels/prostate_00.nii.gz'
    mask = load_and_to_array(mask_path)
    data = load_and_to_array(data_path)
    # plt.figure()
    # show_center(data)
    # plt.figure()
    # plt.plot(np.arange(5), np.arange(5)+1)
    # plt.figure()
    # plt.plot(np.arange(5), np.arange(5)**2)

    # multi_show(np.transpose(mask,(2,0,1)), stride=10).matplot3D()
    # multi_show(np.transpose(mask,(0,1,2)), stride=10).matplot3D()
    # multi_show(np.transpose(mask,(1,0,2)), stride=10).matplot3D()

    multi_show(np.transpose(data,(2,0,1,3)), stride=10).matplot3D()
    multi_show(np.transpose(data,(0,1,2,3)), stride=10).matplot3D()
    multi_show(np.transpose(data,(1,0,2,3)), stride=10).matplot3D()


    image_single_2d = get_certain_slice(data)

    mask_single_2d = get_certain_slice(mask)

    x,y = aug_test(image_single_2d,mask_single_2d)
    f,ax = plt.subplots(2,2)
    ax[0,0].imshow(image_single_2d)
    ax[0,1].imshow(x[0])
    ax[1,0].imshow(mask_single_2d)
    ax[1,1].imshow(y[0])
    plt.figure()
    plt.imshow(x[0][0]-image_single_2d)


    def show_tvtk():
        from tvtk.api import tvtk
        import numpy as np
        from mayavi import mlab
        X, Y, Z = np.mgrid[-10:10:100j, -10:10:100j, -10:10:100j]
        data = np.sin(X * Y * Z) / (X * Y * Z)
        i = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
        i.point_data.scalars = data.ravel()
        i.point_data.scalars.name = 'scalars'
        i.dimensions = data.shape
        mlab.pipeline.surface(i)
        mlab.colorbar(orientation='vertical')
        mlab.show()
    
    
    from nilearn import plotting
    
    # plotting.plot_stat_map(data_path)