import nibabel as nib
import os


def show_dim(distance_root, distance_root2=None):
    for i in os.listdir(distance_root):
        if i.startswith('.'): continue

        path = os.path.join(distance_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        # distance = img_func.get_fdata()
        # print(f'{i} dim = {distance.shape}')
        shape1 = img_func.header.get_data_shape()
        print(f'{i} dim = {shape1}')

        if distance_root2 is not None:
            path = os.path.join(distance_root2, i)
            img_func = nib.load(path)
            shape2 = img_func.header.get_data_shape()
            assert shape1 == shape2


if __name__ == '__main__':
    distance_root = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecAll/val/weight_map_distances_minus_threshold'
    # show_dim(distance_root)

    #
    distance_root2 = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecAll/val/images'
    show_dim(distance_root, distance_root2)
    # # show_dim(distance_root)
    #
    # distance_root2 = '/Users/px/Downloads/MultiPlanarUNet/data_folder/jawDecLabelsAll/weight_map_distances_minus_threshold'
    #
    # show_dim(distance_root2)
    #
    # show_dim(distance_root, distance_root2)
