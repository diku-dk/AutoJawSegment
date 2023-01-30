from mpunet.hyperparameters import YAMLHParams
from mpunet.bin.train import get_logger

import os
os.chdir('/Users/px/GoogleDrive/MultiPlanarUNet')

project_dir = './my_project'
logger = get_logger(project_dir, True)
logger("Fitting model in path:\n%s" % project_dir)

hparams = YAMLHParams(project_dir + "/train_hparams.yaml", logger=logger)

for key, value in hparams.items():
    print(f'{key}: {value}')

from mpunet.preprocessing import get_preprocessing_func
func = get_preprocessing_func(hparams["build"].get("model_class_name"))

from mpunet.preprocessing import data_preparation_funcs as dpf
func = dpf.prepare_for_multi_view_unet

hparams['fit']['flatten_y'] = True
from mpunet.preprocessing.data_preparation_funcs import _base_loader_func
train_queue, val_queue, logger, auditor = _base_loader_func(hparams, just_one=False,
                                                            no_val=False, logger=False, mtype='2d')

# hparams['fit']['max_loaded'] = args.max_loaded_images
# hparams['fit']['num_access'] = args.num_access
# train, val, hparams = func(hparams=hparams,
#                            logger=logger,
#                            just_one=False,
#                            no_val=False,
#                            continue_training=False,
#                            base_path=project_dir)