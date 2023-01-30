from mpunet.logging import ScreenLogger
import os


def init_model(build_hparams, logger=None):
    from mpunet import models
    logger = logger or ScreenLogger()

    # Build new model of the specified type
    cls_name = build_hparams["model_class_name"]
    logger("Creating new model of type '%s'" % cls_name)

    return models.__dict__[cls_name](logger=logger, **build_hparams)


def model_initializer(hparams, continue_training, project_dir,
                      initialize_from=None, logger=None,
                      only_train_last_layer=False,
                      only_train_decoder=False,
                      transfer_last_layer=False,
                      ):
    logger = logger or ScreenLogger()

    # Init model
    model = init_model(hparams["build"], logger)

    if continue_training:
        if initialize_from:
            raise ValueError("Failed to initialize model with both "
                             "continue_training and initialize_from set.")
        from mpunet.utils import get_last_model, get_lr_at_epoch, \
                                          clear_csv_after_epoch, get_last_epoch
        model_path, epoch = get_last_model(os.path.join(project_dir, "model"))
        if model_path:
            cls_name = hparams["build"]["model_class_name"]
            load_by_name = not ('deeplab' in cls_name.lower())

            model.load_weights(model_path,
                               by_name=False
                               )

            model_name = os.path.split(model_path)[-1]
        else:
            model_name = "<No model found>"
        csv_path = os.path.join(project_dir, "logs", "training.csv")
        if epoch == 0:
            epoch = get_last_epoch(csv_path)
        else:
            if epoch is None:
                epoch = 0
            clear_csv_after_epoch(epoch, csv_path)
        hparams["fit"]["init_epoch"] = epoch+1

        # Get the LR at the continued epoch
        lr, name = get_lr_at_epoch(epoch, os.path.join(project_dir, "logs"))
        if lr:
            hparams["fit"]["optimizer_kwargs"][name] = lr

        logger("[NOTICE] Training continues from:\n"
               "Model: %s\n"
               "Epoch: %i\n"
               "LR:    %s" % (model_name, epoch, lr))
    else:
        hparams["fit"]["init_epoch"] = 0
        if initialize_from:
            cls_name = hparams["build"]["model_class_name"]
            if cls_name == 'DeepLabV3Plus':
                model = load_weights_for_deepLab(model, weights=initialize_from)
            else:
                model.load_weights(initialize_from, by_name=not transfer_last_layer)

        #    model.save('./mm_Unet.h5')
            #    model = keras.models.load_model('path/to/location')
        # for i, weights in enumerate(weights_list[:-2]):
        #     model.layers[i].set_weights(weights)

            logger("[NOTICE] Initializing parameters from:\n"
                   "{}".format(initialize_from))

            # for i, weights in enumerate(weights_list[:-2]):
            #     model.layers[i].set_weights(weights)

    if only_train_decoder:
        import pdb
        # pdb.set_trace()

        for layer in model.layers:
            if "encoder" in layer.name or 'bottom' in layer.name:
                layer.trainable = False

    if only_train_last_layer:
        for layer in model.layers[:-2]:
            layer.trainable = False

        # model.tranable = False

        for layer in model.layers[:-2]:
            layer.trainable = False

        for layer in model.layers[::-1]:
            if hasattr(layer, 'activation'):
                layer.tranable = True
                break

    print(model.summary())
    print(f'number of parameters: {model.count_params()}')

    return model

def load_weights_for_deepLab(model, weights='pascal_voc', backbone='xception'):
    from tensorflow.python.keras.utils.data_utils import get_file

    WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
    WEIGHTS_PATH_MOBILE = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5"
    WEIGHTS_PATH_X_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5"
    WEIGHTS_PATH_MOBILE_CS = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.2/deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5"

    if weights == 'pascal_voc':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_MOBILE,
                                    cache_subdir='models')
        model.load_weights(weights_path
                           , by_name=True
                           )
    elif weights == 'cityscapes':
        if backbone == 'xception':
            weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_X_CS,
                                    cache_subdir='models')
        else:
            weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
                                    WEIGHTS_PATH_MOBILE_CS,
                                    cache_subdir='models')
        model.load_weights(weights_path
                           , by_name=True
                           )
        return model