from mpunet.hyperparameters import YAMLHParams
import os
# Check for project dir
project_dir = os.path.abspath('./')

hparams = YAMLHParams(project_dir + "/train_hparams.yaml")

global mask_weight

try:
    mask_weight = hparams['mask_weight']
except:
    mask_weight = False

print(f'mask_weight = {mask_weight}')
# mask_weight = True
