# Crop labels?

# Read in hyperparameters from YAML file
from mpunet.hyperparameters import YAMLHParams

from argparse import ArgumentParser
import os
from train import get_argparser, validate_args, validate_hparams, \
    remove_previous_session, validate_project_dir, get_logger, get_gpu_monitor

import tensorflow as tf

project_dir = '/Users/px/GoogleDrive/MultiPlanarUNet/binary_jaw_project/'
overwrite = True

# Check for project dir
project_dir = os.path.abspath(project_dir)
validate_project_dir(project_dir)
os.chdir(project_dir)

# Get project folder and remove previous session if --overwrite
if overwrite:
    remove_previous_session(project_dir)

# Get logger object. Overwrites previous logfile if args.continue_training
logger = get_logger(project_dir, False)
logger("Fitting model in path:\n%s" % project_dir)

# Start GPU monitor process, if num_GPUs > 0
gpu_mon = get_gpu_monitor(1, logger)

hparams = YAMLHParams(project_dir + "/train_hparams.yaml", logger=logger)
validate_hparams(hparams)

from mpunet.preprocessing import get_preprocessing_func

func = get_preprocessing_func(hparams["build"].get("model_class_name"))
hparams['fit']['flatten_y'] = True
hparams['fit']['max_loaded'] = None
hparams['fit']['num_access'] = 50
train, val, hparams = func(hparams=hparams,
                           logger=logger,
                           just_one=False,
                           no_val=False,
                           continue_training=False,
                           base_path=project_dir)

model = get_model(project_dir=project_dir, train_seq=train,
                  hparams=hparams, logger=logger, args=args)

if hasattr(model, "label_crop"):
    train.label_crop = self.model.label_crop
    if val is not None:
        val.label_crop = self.model.label_crop


from mpunet.utils.plotting import save_images

im_path = os.path.join(logger.base_path, "images")
save_images(train, val, im_path, logger)

dtypes, shapes = list(zip(*map(lambda x: (x.dtype, x.shape), train[0])))

train = tf.data.Dataset.from_generator(train, dtypes, shapes)

if __name__ == "__main__":
    #sb = ['--num_GPUs 1']
    entry_func()
