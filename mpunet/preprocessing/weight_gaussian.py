from weight_map import *

if __name__ == '__main__':

    sigma = 5

    images_root = '../../data_folder/jawDecLabelsAll'

    label_root = os.path.join(images_root, 'labels_indiv_teeth')

    weight_root = os.path.join(images_root, f'weight_maps_no_morph_{sigma}')

    morph = False

    ignore_computed = True

    if not os.path.exists(weight_root):
        os.mkdir(weight_root)

    for i in os.listdir(label_root):
        if i.startswith('.'): continue

        save_path = os.path.join(weight_root, i)

        if ignore_computed and os.path.exists(save_path): continue

        path = os.path.join(label_root, i)
        img_func = nib.load(path)
        affine = img_func.affine
        img_data = img_func.get_fdata()

        img_data = np.squeeze(img_data)

        if morph:
            img_data = binary_erosion_label(img_data, radius=2)

        img_data = img_data.astype(np.float32)

        weight_map = unet_weight_map3D(img_data, w0=20 if morph else 10, sigma=sigma)
        weight_map = weight_map.astype(np.float32)
        ni_img = nib.Nifti1Image(weight_map, affine)
        nib.save(ni_img, save_path)

