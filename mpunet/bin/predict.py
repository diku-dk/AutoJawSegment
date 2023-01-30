"""
Prediction/evaluation script

Mathias Perslev & Peidi XuMarch 2018
"""

import os
import numpy as np
import nibabel as nib

from mpunet.utils.utils import (create_folders, get_best_model,
                                pred_to_class, await_PIDs)
from mpunet.logging.log_results import save_all
from mpunet.evaluate.metrics import dice_all
from argparse import ArgumentParser


def get_argparser():
    parser = ArgumentParser(description='Predict using a mpunet model.')
    parser.add_argument("--project_dir", type=str, default="./",
                        help='Path to mpunet project folder')
    parser.add_argument("-f", help="Predict on a single file")
    parser.add_argument("-l", help="Optional single label file to use with -f")
    parser.add_argument("--dataset", type=str, default="test",
                        help="Which dataset of those stored in the hparams "
                             "file the evaluation should be performed on. "
                             "Has no effect if a single file is specified "
                             "with -f.")
    parser.add_argument("--out_dir", type=str, default="predictions",
                        help="Output folder to store results")
    parser.add_argument("--num_GPUs", type=int, default=1,
                        help="Number of GPUs to use for this job")
    parser.add_argument("--sum_fusion", action="store_true",
                        help="Fuse the mutliple segmentation volumes into one"
                             " by summing over the probability axis instead "
                             "of applying a learned fusion model.")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous results at the output folder')
    parser.add_argument("--no_eval", action="store_true",
                        help="Perform no evaluation of the prediction performance. "
                             "No label files loaded when this flag applies.")
    parser.add_argument("--eval_prob", type=float, default=1.0,
                        help="Perform evaluation on only a fraction of the"
                             " computed views (to speed up run-time). OBS: "
                             "always performs evaluation on the combined "
                             "predictions.")
    parser.add_argument("--force_GPU", type=str, default="")
    parser.add_argument("--save_input_files", action="store_true",
                        help="Save in addition to the predicted volume the "
                             "input image and label files to the output dir)")
    parser.add_argument("--no_argmax", action="store_true",
                        help="Do not argmax prediction volume prior to save.")

    parser.add_argument("--no_softmax", action="store_true",
                        help="Do not argmax prediction volume prior to save.")

    parser.add_argument("--save_single_class", type=int, default=-1,
                        help="Do not argmax prediction volume prior to save.")

    parser.add_argument("--on_val", action="store_true",
                        help="Evaluate on the validation set instead of test")
    parser.add_argument("--wait_for", type=str, default="",
                        help="Waiting for PID to terminate before starting "
                             "training process.")
    parser.add_argument("--continue", action="store_true",
                        help="Continue from a previsous, non-finished "
                             "prediction session at 'out_dir'.")

    parser.add_argument("--binary_sum", action="store_true",
                        help='binary_sum')

    parser.add_argument("--num_extra_planes", type=int, default=0,
                        help="num_extra_planes")

    parser.add_argument("--extra_bound", type=int, default=0,
                        help="num_extra_planes")

    parser.add_argument("--by_radius", action='store_true')

    parser.add_argument("--plane_offset", type=int, default=43,
                        help="plane_offset")

    parser.add_argument("--fuse_batch_size", type=int, default=10 ** 4,
                        help="fuse_batch_size")

    parser.add_argument("--delete_fusion_after", action='store_true')

    parser.add_argument("--only_save_to_disk", action='store_true')

    parser.add_argument("--fusion_save_to_disk", action='store_true')

    parser.add_argument("--only_load_from_disk", action='store_true')

    parser.add_argument("--ccd", action='store_true')
    parser.add_argument("--ccd_portion", type=float, default=0.01,
                        help="fuse_batch_size")

    parser.add_argument("--set_memory_growth", action='store_true',
                        help="build_resnet_connection")

    parser.add_argument("--save_slices", action='store_true',
                        help="build_resnet_connection")

    parser.add_argument("--save_per_view", action='store_true',
                        help="build_resnet_connection")

    parser.add_argument("--predict_batch_size", type=int, default=0,
                        help="build_resnet_connection")

    parser.add_argument("--dim_for_predict", type=int, default=0,
                        help="build_resnet_connection")

    parser.add_argument("--init_filters", type=int, default=64,
                        help="init_filters")

    parser.add_argument("--use_diag_dim", action='store_true')

    parser.add_argument("--single_task", type=int, default=-1,
                        help="build_resnet_connection")

    parser.add_argument("--binary_threshold", type=float, default=0.5,
                        help="build_resnet_connection")

    parser.add_argument("--n_chunks", type=int, default=1,
                        help="build_resnet_connection")

    parser.add_argument("--simple_add", action='store_true',
                        help="build_resnet_connection")

    return parser


def mkdir_safe(p):
    if not os.path.exists(p):
        os.makedirs(p)


def validate_folders(base_dir, out_dir, overwrite, _continue):
    """
    TODO
    """
    # Check base (model) dir contains required files
    must_exist = ("train_hparams.yaml", "views.npz",
                  "model")
    for p in must_exist:
        p = os.path.join(base_dir, p)
        if not os.path.exists(p):
            from sys import exit
            print("[*] Invalid mpunet project folder: '%s'"
                  "\n    Needed file/folder '%s' not found." % (base_dir, p))
            exit(0)

    # Check if output fo
    #
    # lder already exists
    if not (overwrite or _continue) and os.path.exists(out_dir):
        from sys import exit
        print("[*] Output directory already exists at: '%s'"
              "\n  Use --overwrite to overwrite or --continue to continue" % out_dir)
        exit(0)
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)


def save_nii_files(merged, image_pair, nii_res_dir, save_input_files, per_view=''):
    """
    TODO
    """
    # Extract data if nii files

    # import pdb; pdb.set_trace()

    try:
        merged = merged.get_data()
    except AttributeError:
        merged = nib.Nifti1Image(merged, affine=image_pair.affine)
    volumes = [merged, image_pair.image_obj, image_pair.labels_obj]
    labels = [f"{image_pair.identifier + per_view}_PRED.nii.gz",
              f"{image_pair.identifier + per_view}_IMAGE.nii.gz",
              f"{image_pair.identifier + per_view}_LABELS.nii.gz"]
    if not save_input_files:
        volumes = volumes[:1]
        labels = labels[:1]
        p = os.path.abspath(nii_res_dir)  # Save file directly in nii_res_dir
    else:
        # Create sub-folder under nii_res_dir
        p = os.path.join(nii_res_dir, image_pair.identifier)
    create_folders(p)

    for nii, fname in zip(volumes, labels):
        try:
            # nib.save(nii, "%s/%s" % (p, fname))
            nib.save(nii, os.path.join(p, fname))

        except AttributeError:
            # No labels file?
            pass


def remove_already_predicted(all_images, out_dir):
    """
    TODO
    """
    nii_dir = os.path.join(out_dir, "nii_files")
    already_pred = [i.replace("_PRED", "").split(".")[0]
                    for i in filter(None, os.listdir(nii_dir))]
    print("[OBS] Not predicting on images: {} "
          "(--continue mode)".format(already_pred))
    return {k: v for k, v in all_images.items() if k not in already_pred}


def load_hparams(base_dir):
    """
    TODO
    """
    from mpunet.hyperparameters import YAMLHParams
    return YAMLHParams(os.path.join(base_dir, "train_hparams.yaml"))


def set_test_set(hparams, dataset):
    """
    TODO
    """
    hparams['test_dataset'] = hparams[dataset.strip("_dataset") + "_dataset"]


def set_gpu_vis(args):
    """
    TODO
    """
    force_gpu = args.force_GPU
    if not force_gpu:
        # Wait for free GPU
        from mpunet.utils import await_and_set_free_gpu
        await_and_set_free_gpu(N=args.num_GPUs, sleep_seconds=120)
        num_GPUs = args.num_GPUs
    else:
        from mpunet.utils import set_gpu
        set_gpu(force_gpu)
        num_GPUs = len(force_gpu.split(","))
    return num_GPUs


def get_image_pair_loader(args, hparams, out_dir):
    """
    TODO
    """
    from mpunet.image import ImagePairLoader, ImagePair
    if not args.f:
        # No single file was specified with -f flag, load the desired dataset
        dataset = args.dataset.replace("_data", "") + "_data"
        image_pair_loader = ImagePairLoader(predict_mode=args.no_eval,
                                            **hparams[dataset])
    else:
        predict_mode = not bool(args.l)
        image_pair_loader = ImagePairLoader(predict_mode=predict_mode,
                                            initialize_empty=True,
                                            )
        image_pair_loader.add_image(ImagePair(args.f, args.l))

    # Put image pairs into a dict and remove from image_pair_loader to gain
    # more control with garbage collection
    image_pair_dict = {image.identifier: image for image in image_pair_loader.images}
    if vars(args)["continue"]:
        # Remove images that were already predicted
        image_pair_dict = remove_already_predicted(image_pair_dict, out_dir)
    return image_pair_loader, image_pair_dict


def get_results_dicts(out_dir, views, image_pairs_dict, n_classes, _continue):
    """
    TODO
    """
    from mpunet.logging import init_result_dicts, save_all, load_result_dicts
    if _continue:
        csv_dir = os.path.join(out_dir, "csv")
        results, detailed_res = load_result_dicts(csv_dir=csv_dir, views=views)
    else:
        # Prepare dictionary to store results in pd df
        results, detailed_res = init_result_dicts(views, image_pairs_dict, n_classes)
    # Save to check correct format
    save_all(results, detailed_res, out_dir)
    return results, detailed_res


def get_model(project_dir, build_hparams):
    """
    TODO
    """
    from mpunet.models.model_init import init_model
    model_path = get_best_model(project_dir + "/model")
    weights_name = os.path.splitext(os.path.split(model_path)[1])[0]
    print("\n[*] Loading model weights:\n", model_path)
    import tensorflow as tf


    strategy = tf.distribute.MirroredStrategy()

    #with strategy.scope():
    if True:
        model = init_model(build_hparams)
        print(model.summary())
        # model.load_weights(model_path, by_name=True)
        model.load_weights(model_path,
                           by_name=False
                           )

        return model, weights_name


def get_fusion_model(n_views, n_classes, project_dir, weights_name):
    """
    TODO
    """
    from mpunet.models import FusionModel
    fm = FusionModel(n_inputs=n_views, n_classes=n_classes)
    # Load fusion weights
    weights = project_dir + "/model/fusion_weights/%s_fusion_" \
                            "weights.h5" % weights_name
    print("\n[*] Loading fusion model weights:\n", weights)
    fm.load_weights(weights)
    print("\nLoaded weights:\n\n%s\n%s\n---" % tuple(
        fm.layers[-1].get_weights()))
    return fm


def evaluate(pred, true, n_classes, ignore_zero=False):
    """
    TODO
    """
    pred = pred_to_class(pred, img_dims=3, has_batch_dim=False)
    return dice_all(y_true=true,
                    y_pred=pred,
                    ignore_zero=ignore_zero,
                    n_classes=n_classes,
                    skip_if_no_y=False)


def _per_view_evaluation(image_id, pred, true, mapped_pred, mapped_true, view,
                         n_classes, results, per_view_results, out_dir, args):
    """
    TODO
    """
    if np.random.rand() > args.eval_prob:
        print("Skipping evaluation for view %s... "
              "(eval_prob=%.3f)" % (view, args.eval_prob))
        return

    # Evaluate the raw view performance
    view_dices = evaluate(pred, true, n_classes)
    mapped_dices = evaluate(mapped_pred, mapped_true, n_classes)
    mean_dice = mapped_dices[~np.isnan(mapped_dices)][1:].mean()

    # Print dice scores
    print("View dice scores:   ", view_dices)
    print("Mapped dice scores: ", mapped_dices)
    print("Mean dice (n=%i): " % (len(mapped_dices) - 1), mean_dice)

    # Add to results
    results.loc[image_id, str(view)] = mean_dice
    per_view_results[str(view)][image_id] = mapped_dices[1:]

    # Overwrite with so-far results
    save_all(results, per_view_results, out_dir)


def _merged_eval(image_id, pred, true, n_classes, results,
                 per_view_results, out_dir):
    """
    TODO
    """
    # Calculate combined prediction dice
    dices = evaluate(pred, true, n_classes, ignore_zero=True)
    mean_dice = dices[~np.isnan(dices)].mean()
    per_view_results["MJ"][image_id] = dices

    print("Combined dices: ", dices)
    print("Combined mean dice: ", mean_dice)
    results.loc[image_id, "MJ"] = mean_dice

    # Overwrite with so-far results
    save_all(results, per_view_results, out_dir)


def _multi_view_predict_on(image_pair, seq, model, views, results,
                           per_view_results, out_dir, args, n_image=0, save_to_disk=False,
                           sub_task=False, single_task=-1):
    """
    TODO
    """
    from mpunet.utils.fusion import predict_volume, map_real_space_pred
    from mpunet.interpolation.sample_grid import get_voxel_grid_real_space

    # Prepare tensor to store combined prediction
    d = image_pair.image.shape[:-1]

    num_task = 2 if sub_task else 1

    print(f'image shape {d}')

    # Get voxel grid in real space
    voxel_grid_real_space = get_voxel_grid_real_space(shape=image_pair.shape[:-1],
                                                      vox_to_real_affine=image_pair.affine[:-1, :-1])

    if save_to_disk:
        basedir = os.path.abspath(args.project_dir)
        fusion_dataset_test = os.path.join(basedir, 'fusion_dataset_test')

        image_i_fusion_dir = os.path.join(fusion_dataset_test, str(n_image))

        fusion_label_name = 'fusion_label.csv'

    combined = []

    n_classes = 1 if seq.n_classes <= 2 else seq.n_classes

    combined.append(np.empty(
        shape=(len(views), d[0], d[1], d[2], n_classes),
        dtype=np.float32
    ))
    print("Predicting on brain hyper-volume of shape:", combined[0].shape)

    if sub_task:
        if single_task == -1:
            combined.append(np.empty(
                shape=(len(views), d[0], d[1], d[2], 1),
                dtype=np.float32
            ))
            print("Predicting on brain hyper-volume of shape:", combined[1].shape)
        elif single_task == 0:
            pass
        elif single_task == 1:
            combined[0] = np.empty(shape=(len(views), d[0], d[1], d[2], 1),
                                   dtype=np.float32
                                   )


    # Predict for each view
    basedir = os.path.abspath(args.project_dir)
    slices_dir = os.path.join(basedir, 'slices_2D')
    if not os.path.exists(slices_dir):
        os.mkdir(slices_dir)

    for n_view, view in enumerate(views):
        print("\n[*] (%i/%i) View: %s" % (n_view + 1, len(views), view))
        # for each view, predict on all voxels and map the predictions
        # back into the original coordinate system

        # Sample planes from the image at grid_real_space grid
        # in real space (scanner RAS) coordinates.

        n_planes = 'same+' + str(args.num_extra_planes)  # 'same+20' by default

        n_offset = args.plane_offset  # 43 by default
        n_bounds = n_offset if n_offset != 43 else None

        if args.by_radius:
            n_planes = 'by_radius'

        extra_bound = args.extra_bound  # 0
        extra_plane = args.num_extra_planes

        X, y, grid, inv_basis = seq.get_view_from(image_pair, view,
                                                  n_planes=n_planes,
                                                  n_bounds=n_bounds,
                                                  extra_bound=extra_bound,
                                                  extra_plane=extra_plane,

                                                  )

        # Predict on volume using model

        print(f'X shape {X.shape}')

        # .sample_dim, .sample_dim, n_planes， n_channels

        predict_batch_size = seq.batch_size if args.predict_batch_size == 0 else args.predict_batch_size

        pred = predict_volume(model, X, axis=2, batch_size=predict_batch_size, sub_task=sub_task, n_chunks=args.n_chunks)

        if not sub_task:
            pred = [pred]
        else:
            if single_task > -1:
                pred = [pred[single_task]]

        num_task = len(pred)

        for i in range(num_task):
            print(f'pred shape for task {i} = {pred[i].shape}')

            if args.save_slices:
                slices_dir_cur = os.path.join(slices_dir, f'task_{i}_view_{n_view}')
                save_slices_to_disk(np.moveaxis(X, 2, 0), np.moveaxis(pred[i], 2, 0), slices_dir_cur)

            # breakpoint()

            if args.no_eval and i == num_task - 1:
                del X, y

            # Map the real space coordiante predictions to nearest
            # real space coordinates defined on voxel grid
            mapped_pred = map_real_space_pred(pred[i], grid, inv_basis,
                                              voxel_grid_real_space,
                                              method="nearest")

            print(f'mapped_pred for task {i} shape = {mapped_pred.shape}')

            if save_to_disk:
                points_name = f'task_{i}_view_{n_view}'
                points_path = os.path.join(image_i_fusion_dir, points_name)

                # np.save(points_path, mapped_pred.astype(np.float32))

                np.savez_compressed(points_path, mapped_pred.astype(np.float32))

                shapes = mapped_pred.shape[0]

            else:
                combined[i][n_view] = mapped_pred

            if args.save_per_view:
                # Save combined prediction volume as .nii file
                print("Saving .nii files per view ...")

                basedir = os.path.abspath(args.project_dir)
                basedir = os.path.join(basedir, f'task_{i}')
                per_view_res_dir = os.path.join(basedir, 'per_view_res')
                if not os.path.exists(per_view_res_dir):
                    os.mkdir(per_view_res_dir)

                create_folders(per_view_res_dir, create_deep=True)
                save_nii_files(merged=pred_to_class(mapped_pred.squeeze(), img_dims=3).astype(np.uint8),
                               image_pair=image_pair,
                               nii_res_dir=per_view_res_dir,
                               save_input_files=args.save_input_files,
                               per_view=str(n_view))

            # combined shape: (n_view) +  mapped_pred.shape

            if not args.no_eval:
                _per_view_evaluation(image_id=image_pair.identifier,
                                     pred=pred[i],
                                     true=y,
                                     mapped_pred=mapped_pred,
                                     mapped_true=image_pair.labels,
                                     view=view,
                                     n_classes=seq.n_classes,
                                     results=results,
                                     per_view_results=per_view_results,
                                     out_dir=out_dir,
                                     args=args)
                del X, y

            del mapped_pred,
            pred[i] = None
        del pred, grid, inv_basis
        try:
            del X
        except:
            pass

    if save_to_disk:
        return shapes
    else:
        return combined



def _multi_view_predict_on_sum_fusion(image_pair, seq, model, views, results,
                           per_view_results, out_dir, args, n_image=0, save_to_disk=False,
                           sub_task=False, single_task=-1):
    """
    TODO
    """
    from mpunet.utils.fusion import predict_volume, map_real_space_pred
    from mpunet.interpolation.sample_grid import get_voxel_grid_real_space

    # Prepare tensor to store combined prediction
    d = image_pair.image.shape[:-1]

    num_task = 2 if sub_task else 1

    print(f'image shape {d}')

    # Get voxel grid in real space
    voxel_grid_real_space = get_voxel_grid_real_space(shape=image_pair.shape[:-1],
                                                      vox_to_real_affine=image_pair.affine[:-1, :-1])


    combined = []

    n_classes = 1 if seq.n_classes <= 2 else seq.n_classes

    combined.append(np.empty(
        shape=(d[0], d[1], d[2], n_classes),
        dtype=np.float32
    ))
    print("Predicting on brain hyper-volume of shape:", combined[0].shape)

    if sub_task:
        if single_task == -1:
            combined.append(np.empty(
                shape=(d[0], d[1], d[2], 1),
                dtype=np.float32
            ))
            print("Predicting on brain hyper-volume of shape:", combined[1].shape)
        elif single_task == 0:
            pass
        elif single_task == 1:
            combined[0] = np.empty(shape=(d[0], d[1], d[2], 1),
                                   dtype=np.float32
                                   )


    # Predict for each view
    basedir = os.path.abspath(args.project_dir)
    slices_dir = os.path.join(basedir, 'slices_2D')
    if not os.path.exists(slices_dir):
        os.mkdir(slices_dir)

    for n_view, view in enumerate(views):
        print("\n[*] (%i/%i) View: %s" % (n_view + 1, len(views), view))
        # for each view, predict on all voxels and map the predictions
        # back into the original coordinate system

        # Sample planes from the image at grid_real_space grid
        # in real space (scanner RAS) coordinates.

        n_planes = 'same+' + str(args.num_extra_planes)  # 'same+20' by default

        n_offset = args.plane_offset  # 43 by default
        n_bounds = n_offset if n_offset != 43 else None

        if args.by_radius:
            n_planes = 'by_radius'

        extra_bound = args.extra_bound  # 0
        extra_plane = args.num_extra_planes

        X, y, grid, inv_basis = seq.get_view_from(image_pair, view,
                                                  n_planes=n_planes,
                                                  n_bounds=n_bounds,
                                                  extra_bound=extra_bound,
                                                  extra_plane=extra_plane,

                                                  )

        # Predict on volume using model

        print(f'X shape {X.shape}')

        # .sample_dim, .sample_dim, n_planes， n_channels

        predict_batch_size = seq.batch_size if args.predict_batch_size == 0 else args.predict_batch_size

        pred = predict_volume(model, X, axis=2, batch_size=predict_batch_size, sub_task=sub_task, n_chunks=args.n_chunks)

        if not sub_task:
            pred = [pred]
        else:
            if single_task > -1:
                pred = [pred[single_task]]

        num_task = len(pred)

        for i in range(num_task):
            print(f'pred shape for task {i} = {pred[i].shape}')

            if args.save_slices:
                slices_dir_cur = os.path.join(slices_dir, f'task_{i}_view_{n_view}')
                save_slices_to_disk(np.moveaxis(X, 2, 0), np.moveaxis(pred[i], 2, 0), slices_dir_cur)

            # breakpoint()

            if args.no_eval and i == num_task - 1:
                del X, y

            # Map the real space coordiante predictions to nearest
            # real space coordinates defined on voxel grid
            mapped_pred = map_real_space_pred(pred[i], grid, inv_basis,
                                              voxel_grid_real_space,
                                              method="nearest")

            print(f'mapped_pred for task {i} shape = {mapped_pred.shape}')

            combined[i] += mapped_pred

            if args.save_per_view:
                # Save combined prediction volume as .nii file
                print("Saving .nii files per view ...")

                basedir = os.path.abspath(args.project_dir)
                basedir = os.path.join(basedir, f'task_{i}')
                per_view_res_dir = os.path.join(basedir, 'per_view_res')
                if not os.path.exists(per_view_res_dir):
                    os.mkdir(per_view_res_dir)

                create_folders(per_view_res_dir, create_deep=True)
                save_nii_files(merged=pred_to_class(mapped_pred.squeeze(), img_dims=3).astype(np.uint8),
                               image_pair=image_pair,
                               nii_res_dir=per_view_res_dir,
                               save_input_files=args.save_input_files,
                               per_view=str(n_view))

            # combined shape: (n_view) +  mapped_pred.shape

            if not args.no_eval:
                _per_view_evaluation(image_id=image_pair.identifier,
                                     pred=pred[i],
                                     true=y,
                                     mapped_pred=mapped_pred,
                                     mapped_true=image_pair.labels,
                                     view=view,
                                     n_classes=seq.n_classes,
                                     results=results,
                                     per_view_results=per_view_results,
                                     out_dir=out_dir,
                                     args=args)
                del X, y

            del mapped_pred,
            pred[i] = None
        del pred, grid, inv_basis
        try:
            del X
        except:
            pass


    return combined


def merge_multi_view_preds(multi_view_preds, fusion_model, args, softmax=True):
    """
    TODO
    """

    # multi_view_preds： n_view * dim1 * dim2 * dim3 * n_class

    fm = fusion_model
    if not args.sum_fusion:
        fuse_batch_size = args.fuse_batch_size  # 10**4 by default

        # Combine predictions across views using Fusion model
        print("\nFusing views (fusion model)...")
        d = multi_view_preds.shape
        multi_view_preds = np.moveaxis(multi_view_preds, 0, -2)
        multi_view_preds = multi_view_preds.reshape((-1, fm.n_inputs, fm.n_classes))
        merged = fm.predict(multi_view_preds, batch_size=fuse_batch_size, verbose=1)
        merged = merged.reshape((d[1], d[2], d[3], fm.n_classes))
    else:
        print("\nFusion views (sum)...")
        merged = np.mean(multi_view_preds, axis=0)

    print(f"\nmerged shape{merged.shape}")

    if args.binary_sum:
        from mpunet.utils import pred_to_probabilities
        merged = pred_to_probabilities(merged, img_dims=3, has_batch_dim=False, softmax=softmax)

    merged = merged.squeeze()
    merged_map = pred_to_class(merged, threshold=args.binary_threshold, img_dims=3).astype(np.uint8)
    return merged, merged_map


def merge_preds_sum_fusion(image_i_fusion_dir, args=None):
    # merged = np.empty(shape=(image_len, n_classes),
    #                   dtype=np.float32
    #                   )
    merged = 0
    views_path = list(sorted(os.listdir(image_i_fusion_dir)))
    views_path = [os.path.join(image_i_fusion_dir, f) for f in views_path]
    for i, v_path in enumerate(views_path):
        try:
            merged += np.load(v_path).astype(np.float32)
        except:
            merged += np.load(v_path)['arr_0'].astype(np.float32)

    merged /= len(views_path)

    merged_map = pred_to_class(merged.squeeze(), threshold=args.binary_threshold, img_dims=3).astype(np.uint8)
    return merged, merged_map


def merge_preds_from_disk(n_classes, image_len, fusion_model,
                          d, image_i_fusion_dir, views, args, dtype=None):
    fm = fusion_model

    fuse_batch_size = args.fuse_batch_size  # 10**4 by default

    # multi_view_preds = np.empty(
    #     shape=(len(views), d[0], d[1], d[2], n_classes),
    #     dtype=np.float32
    # )

    merged = np.empty(shape=(image_len, n_classes),
                      dtype=np.float32
                      )

    n_batches = image_len // fuse_batch_size + 1
    ranges = np.array_split(np.arange(0, image_len), n_batches)

    print(f'predict on {n_batches} batches with length {fuse_batch_size} each')

    for iter_num, single_range in enumerate(ranges):
        begin_index = single_range[0]
        end_index = single_range[-1] + 1  #### check if +1 is correct here

        multi_view_preds = np.empty(
            shape=(len(views), len(single_range)
                   , n_classes),
            dtype=np.float32
        )
        views_path = list(sorted(os.listdir(image_i_fusion_dir)))
        views_path = [os.path.join(image_i_fusion_dir, f) for f in views_path]
        for i, v_path in enumerate(views_path):
            multi_view_preds[i] = np.loadtxt(v_path,
                                             max_rows=end_index - begin_index,
                                             skiprows=begin_index).astype(np.float32)
        if not args.sum_fusion:
            if iter_num % 10 == 0:
                print(f'summing on iteration{iter_num}')
            multi_view_preds = np.moveaxis(multi_view_preds, 0, -2)
            # multi_view_preds = multi_view_preds.reshape((-1, fm.n_inputs, fm.n_classes))

            merged[begin_index: end_index] = fm.predict(multi_view_preds, verbose=1)
        else:
            merged[begin_index: end_index] = np.mean(multi_view_preds, axis=0)

    merged = merged.reshape((d[0], d[1], d[2], fm.n_classes))

    # Combine predictions across views using Fusion model
    print("\nFusing views (fusion model)...")

    print(f"\nmerged shape{merged.shape}")

    if args.binary_sum:
        from mpunet.utils import pred_to_probabilities
        merged = pred_to_probabilities(merged, img_dims=3, has_batch_dim=False, softmax=True)

    merged_map = pred_to_class(merged.squeeze(), img_dims=3).astype(np.uint8)
    return merged, merged_map


def run_predictions_and_eval(image_pair_loader, image_pair_dict, model,
                             fusion_model, views, hparams, args, results,
                             per_view_results, out_dir, nii_res_dir):
    """
    TODO
    """
    # Set scaler and bg values
    image_pair_loader.set_scaler_and_bg_values(
        bg_value=hparams.get_from_anywhere('bg_value'),
        scaler=hparams.get_from_anywhere('scaler'),
        compute_now=False
    )

    single_task = args.single_task

    # Init LazyQueue and get its sequencer
    from mpunet.sequences.utils import get_sequence
    seq = get_sequence(data_queue=image_pair_loader,
                       is_validation=True,
                       views=views,
                       **hparams["fit"], **hparams["build"])

    print(f'save only class = {args.save_single_class}')

    print(f'seq batch size: {seq.batch_size}')

    print(f'seq dim: {hparams["build"]["dim"]}')

    # from mpunet.utils.fusion.fuse_and_predict import predict_single
    # o = predict_single(image_pair_loader.images[0], model, hparams)
    # print(o.shape)

    image_ids = sorted(image_pair_dict)
    n_images = len(image_ids)

    if args.fusion_save_to_disk:
        basedir = os.path.abspath(args.project_dir)
        fusion_dataset_test = os.path.join(basedir, 'fusion_dataset_test')
        mkdir_safe(fusion_dataset_test)
        for i in range(n_images):
            image_i_fusion_dir = os.path.join(fusion_dataset_test, str(i))
            mkdir_safe(image_i_fusion_dir)

    for n_image, image_id in enumerate(image_ids):
        print("\n[*] (%i/%s) Running on: %s" % (n_image + 1, n_images, image_id))

        with seq.image_pair_queue.get_image_by_id(image_id) as image_pair:
            # Get prediction through all views

            real_shape = image_pair.real_shape

            print(f'real shape is {real_shape}')

            real_space_span = np.max(real_shape)

            real_space_span = np.linalg.norm(sorted(real_shape[1:]))

            print(f'real shape for all = {seq.real_space_span}')
            print(f'real shape for current = {real_space_span}')

            seq.real_space_span = max(real_space_span, seq.real_space_span)

            if not args.only_load_from_disk:
                print('getting predicctions')

                try:
                    sub_task = hparams["fit"]['sub_task']
                except:
                    sub_task = False
                multi_view_preds = _multi_view_predict_on(
                    image_pair=image_pair,
                    seq=seq,
                    model=model,
                    views=views,
                    results=results,
                    per_view_results=per_view_results,
                    out_dir=out_dir,
                    args=args,
                    n_image=n_image,
                    save_to_disk=args.fusion_save_to_disk,
                    sub_task=sub_task,
                    single_task=single_task
                )

            else:
                image_len = 400 * 400 * 280

            if args.only_save_to_disk:
                return

            if args.fusion_save_to_disk:
                d = image_pair.image.shape[:-1]

                basedir = os.path.abspath(args.project_dir)
                fusion_dataset_test = os.path.join(basedir, 'fusion_dataset_test')
                image_i_fusion_dir = os.path.join(fusion_dataset_test, str(n_image))

                print('loading from disk')

                if args.sum_fusion:
                    merged, merged_map = merge_preds_sum_fusion(image_i_fusion_dir=image_i_fusion_dir, args=args)

                else:
                    # Merge the multi view predictions into a final segmentation
                    merged, merged_map = merge_preds_from_disk(n_classes=seq.n_classes,
                                                               image_len=image_len,
                                                               fusion_model=fusion_model,
                                                               d=d,
                                                               image_i_fusion_dir=image_i_fusion_dir,
                                                               views=views, args=args)

                if os.path.exists(fusion_dataset_test):
                    import shutil
                    shutil.rmtree(image_i_fusion_dir)

                # Save combined prediction volume as .nii file
                print("Saving .nii files...")

                # breakpoint()

                if args.no_argmax:
                    merged_map = merged.squeeze().astype(np.float32)
                    if args.save_single_class > -1:
                        if args.save_single_class > 10:
                            class_index = str(args.save_single_class)
                            merged_map = merged_map[..., int(class_index[0])] + merged_map[..., int(class_index[1])]
                        else:
                            merged_map = merged_map[..., args.save_single_class]

                        # merged_map = np.expand_dims(merged_map, axis=-1)

                save_nii_files(merged=merged_map,
                               image_pair=image_pair,
                               nii_res_dir=nii_res_dir,
                               save_input_files=args.save_input_files)

                try:
                    del merged
                except:
                    merged = None
                try:
                    del merged_map
                except:
                    merged_map = None

            else:
                # Merge the multi view predictions into a final segmentation
                for i in range(len(multi_view_preds)):

                    softmax = (len(multi_view_preds)) > 1 and i == 0 or single_task == 0 and args.binary_sum

                    merged, merged_map = merge_multi_view_preds(multi_view_preds[i],
                                                                fusion_model, args,
                                                                softmax=softmax)
                    if args.ccd:
                        from mpunet.postprocessing import connected_component_3D
                        # merged_map = connected_component_3D(merged_map, portion_foreground=args.ccd_portion)
                        from mpunet.postprocessing import symmetric_separator
                        merged_map = symmetric_separator(merged_map, portion_foreground=args.ccd_portion)

                    if not args.no_eval:
                        _merged_eval(
                            image_id=image_id,
                            pred=merged_map,
                            true=image_pair.labels,
                            n_classes=hparams["build"]["n_classes"],
                            results=results,
                            per_view_results=per_view_results,
                            out_dir=out_dir
                        )

                    # del multi_view_preds
                    if i == len(multi_view_preds) - 1:
                        del multi_view_preds
                    else:
                        multi_view_preds[i] = 0

                    # print(f'**************binary_sum {args.binary_sum} **************')
                    # print(f'************** merged.shape {merged.shape} **************')
                    #
                    # if args.binary_sum:
                    #     merged = merged[..., 0]

                    # Save combined prediction volume as .nii file
                    print("Saving .nii files...")

                    # breakpoint()

                    if args.no_argmax:
                        merged_map = merged.squeeze().astype(np.float32)
                        if args.save_single_class > -1:
                            if args.save_single_class > 10:
                                class_index = str(args.save_single_class)
                                merged_map = merged_map[..., int(class_index[0])] + merged_map[..., int(class_index[1])]
                            else:
                                merged_map = merged_map[..., args.save_single_class]

                    # breakpoint()
                    if i > 0 and single_task != 0:
                        merged_map = merged.squeeze().astype(np.float32)

                    save_nii_files(merged=merged_map,
                                   image_pair=image_pair,
                                   nii_res_dir=nii_res_dir + f'_task_{i}',
                                   save_input_files=args.save_input_files,
                                   )

                    try:
                        del merged

                    except:
                        merged = None
                    try:
                        del merged_map
                    except:
                        merged_map = None
                try:
                    del multi_view_preds
                except:
                    pass



def run_predictions_and_eval_sum_fusion(image_pair_loader, image_pair_dict, model,
                                     fusion_model, views, hparams, args, results,
                                     per_view_results, out_dir, nii_res_dir):

    # Set scaler and bg values
    image_pair_loader.set_scaler_and_bg_values(
        bg_value=hparams.get_from_anywhere('bg_value'),
        scaler=hparams.get_from_anywhere('scaler'),
        compute_now=False
    )

    single_task = args.single_task

    # Init LazyQueue and get its sequencer
    from mpunet.sequences.utils import get_sequence
    seq = get_sequence(data_queue=image_pair_loader,
                       is_validation=True,
                       views=views,
                       **hparams["fit"], **hparams["build"])

    print(f'save only class = {args.save_single_class}')

    print(f'seq batch size: {seq.batch_size}')

    print(f'seq dim: {hparams["build"]["dim"]}')


    image_ids = sorted(image_pair_dict)
    n_images = len(image_ids)

    for n_image, image_id in enumerate(image_ids):
        print("\n[*] (%i/%s) Running on: %s" % (n_image + 1, n_images, image_id))

        with seq.image_pair_queue.get_image_by_id(image_id) as image_pair:
            # Get prediction through all views

            real_shape = image_pair.real_shape

            print(f'real shape is {real_shape}')

            real_space_span = np.max(real_shape)

            real_space_span = np.linalg.norm(sorted(real_shape[1:]))

            print(f'real shape for all = {seq.real_space_span}')
            print(f'real shape for current = {real_space_span}')

            seq.real_space_span = max(real_space_span, seq.real_space_span)

            print('getting predicctions')

            try:
                sub_task = hparams["fit"]['sub_task']
            except:
                sub_task = False

            multi_view_preds = _multi_view_predict_on_sum_fusion(
                image_pair=image_pair,
                seq=seq,
                model=model,
                views=views,
                results=results,
                per_view_results=per_view_results,
                out_dir=out_dir,
                args=args,
                n_image=n_image,
                save_to_disk=args.fusion_save_to_disk,
                sub_task=sub_task,
                single_task=single_task
            )


            for i in range(len(multi_view_preds)):

                softmax = (len(multi_view_preds)) > 1 and i == 0 or single_task == 0 and args.binary_sum

                merged_map = pred_to_class(multi_view_preds[i].squeeze()/len(views),
                                           threshold=args.binary_threshold, img_dims=3).astype(np.uint8)


                if not args.no_eval:
                    _merged_eval(
                        image_id=image_id,
                        pred=merged_map,
                        true=image_pair.labels,
                        n_classes=hparams["build"]["n_classes"],
                        results=results,
                        per_view_results=per_view_results,
                        out_dir=out_dir
                    )

                # del multi_view_preds
                if i == len(multi_view_preds) - 1:
                    del multi_view_preds
                else:
                    multi_view_preds[i] = 0


                print("Saving .nii files...")

                # breakpoint()
                #
                # if args.no_argmax:
                #     merged_map = merged.squeeze().astype(np.float32)
                #     if args.save_single_class > -1:
                #         if args.save_single_class > 10:
                #             class_index = str(args.save_single_class)
                #             merged_map = merged_map[..., int(class_index[0])] + merged_map[..., int(class_index[1])]
                #         else:
                #             merged_map = merged_map[..., args.save_single_class]
                #
                # # breakpoint()
                # if i > 0 and single_task != 0:
                #     merged_map = merged.squeeze().astype(np.float32)

                save_nii_files(merged=merged_map,
                               image_pair=image_pair,
                               nii_res_dir=nii_res_dir + f'_task_{i}',
                               save_input_files=args.save_input_files,
                               )


                try:
                    del merged_map
                except:
                    merged_map = None
            try:
                del multi_view_preds
            except:
                pass


def assert_args(args):
    pass


def entry_func(args=None):
    # Get command line arguments
    args = get_argparser().parse_args(args)
    assert_args(args)

    # Get most important paths
    project_dir = os.path.abspath(args.project_dir)
    out_dir = os.path.abspath(args.out_dir)

    # Check if valid dir structures
    validate_folders(project_dir, out_dir,
                     overwrite=args.overwrite,
                     _continue=vars(args)["continue"])
    nii_res_dir = os.path.join(out_dir, "nii_files")
    create_folders(nii_res_dir, create_deep=True)

    # Get settings from YAML file
    hparams = load_hparams(project_dir)

    if args.dim_for_predict > 0:
        hparams['build']['dim'] = args.dim_for_predict

    elif args.use_diag_dim:
        dim = hparams['build']['dim']
        import math
        hparams['build']['dim'] = int(round(dim * math.sqrt(2) / 16) * 16)

    # Get dataset
    image_pair_loader, image_pair_dict = get_image_pair_loader(args, hparams,
                                                               out_dir)

    # Wait for PID to terminate before continuing, if specified
    if args.wait_for:
        await_PIDs(args.wait_for, check_every=120)

    # Set GPU device
    set_gpu_vis(args)

    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices and args.set_memory_growth:
        print(f'**********\n {physical_devices} **********\n')
        # tf.config.gpu.set_per_process_memory_fraction(0.75)
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        # tf.config.experimental.set_virtual_device_configuration(physical_devices[0],
        #                                                         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)])

    # Get views
    views = np.load("%s/views.npz" % project_dir)["arr_0"]
    del hparams['fit']['views']

    # Prepare result dicts
    results, per_view_results = None, None
    if not args.no_eval:
        results, per_view_results = get_results_dicts(out_dir, views,
                                                      image_pair_dict,
                                                      hparams["build"]["n_classes"],
                                                      vars(args)["continue"])

    # Get model and load weights, assign to one or more GPUs

    if args.no_softmax:
        hparams['build']['out_activation'] = 'sigmoid'

    hparams["build"]['init_filters'] = args.init_filters

    model, weights_name = get_model(project_dir, hparams['build'])
    fusion_model = None
    if not args.sum_fusion:
        fusion_model = get_fusion_model(n_views=len(views),
                                        n_classes=hparams["build"]["n_classes"],
                                        project_dir=project_dir,
                                        weights_name=weights_name)

    if not args.simple_add:
        run_predictions_and_eval(
            image_pair_loader=image_pair_loader,
            image_pair_dict=image_pair_dict,
            model=model,
            fusion_model=fusion_model,
            views=views,
            hparams=hparams,
            args=args,
            results=results,
            per_view_results=per_view_results,
            out_dir=out_dir,
            nii_res_dir=nii_res_dir
        )

    else:
        run_predictions_and_eval_sum_fusion(
            image_pair_loader=image_pair_loader,
            image_pair_dict=image_pair_dict,
            model=model,
            fusion_model=fusion_model,
            views=views,
            hparams=hparams,
            args=args,
            results=results,
            per_view_results=per_view_results,
            out_dir=out_dir,
            nii_res_dir=nii_res_dir
        )

    if not args.no_eval:
        # Write final results
        save_all(results, per_view_results, out_dir)


def save_slices_to_disk(X, pred, subdir):
    import matplotlib.pyplot as plt
    from mpunet.utils.plotting import (imshow_with_label_overlay, imshow,
                                       plot_all_training_curves, imshow_weight_map)

    if not os.path.exists(subdir):
        os.mkdir(subdir)

    for i, (im, p) in enumerate(zip(X, pred)):
        try:
            p = p.reshape(p.shape[:-1] + (p.shape[-1],))
        except:
            p = p.reshape(p.shape[:-1] + (1,))

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 6))

        # Imshow pred on ax3
        chnl, axis, slice = imshow_with_label_overlay(ax2, im, p, lab_alpha=1.0,
                                                      # channel=chnl, axis=axis,
                                                      # slice=slice
                                                      )

        if slice is not None:
            # Only for 3D imges
            im = im[slice]
        ax1.imshow(im, cmap="gray")

        # Set labels
        ax1.set_title("Image", size=18)
        ax2.set_title("Predictions", size=18)

        fig.tight_layout()
        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning)
            fig.savefig(os.path.join(subdir, str(i) + ".png"))
        plt.close(fig.number)


def round_to_multiple(num, base):
    return round(num / base) * base


if __name__ == "__main__":
    entry_func()

