from pathlib import Path
from mpunet.hyperparameters import YAMLHParams
from mpunet.bin.train import get_logger
import numpy as np
import os

os.chdir('/Users/px/GoogleDrive/MultiPlanarUNet')

project_dir = './my_project'
logger = get_logger(project_dir, True)
logger("Fitting model in path:\n%s" % project_dir)

hparams = YAMLHParams(project_dir + "/train_hparams.yaml", logger=logger)

for key, value in hparams.items():
    print(f'{key}: {value}')
from mpunet.preprocessing import data_preparation_funcs as dpf
func = dpf.prepare_for_multi_view_unet
from mpunet.image.image_pair_loader import ImagePairLoader
train_data = ImagePairLoader(logger=logger, **hparams["train_data"])
val_data = ImagePairLoader(logger=logger, **hparams["val_data"])
from mpunet.image.queue.utils import get_data_queues

for dataset in (train_data, val_data):
    logger("Preparing dataset {}".format(dataset))
    dataset.set_scaler_and_bg_values(bg_value=hparams.get_from_anywhere('bg_value'),
                                     scaler=hparams.get_from_anywhere('scaler'),
                                     compute_now=False)


max_loaded = hparams['fit'].get('max_loaded')
train_queue, val_queue = get_data_queues(
    train_dataset=train_data,
    val_dataset=val_data,
    train_queue_type='limitation' if max_loaded else "eager",
    val_queue_type='eager',
    max_loaded=max_loaded,
    num_access_before_reload=hparams['fit'].get('num_access'),
    logger=logger
)

view_path = '/Users/px/GoogleDrive/' \
            'MultiPlanarUNet/my_project/views.npz'
views = np.load(view_path)
views_data = views['arr_0']
hparams["fit"]["views"] = views_data
from mpunet.preprocessing.data_preparation_funcs import get_sequencers
train, val = get_sequencers(train_queue, val_queue, logger, hparams)

train.batch_size = 6

single_batch = train[0]

#%%
x.shape
#%%