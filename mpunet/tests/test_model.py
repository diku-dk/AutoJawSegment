import os
import sys
import argparse

current_path = os.path.abspath(__file__)
home_dir = os.path.dirname(os.path.dirname(current_path))
#sys.path.append(home_dir, os.path.join(home_dir, "lib"))
from mpunet.models.unet import UNet
from mpunet.models.models_no_log import UNet3D

my_model = UNet(n_classes=4,img_rows=224,img_cols=224)
print(my_model.summary())


my_model = UNet3D(n_classes=4, dim=224)
print(my_model.summary())
print(my_model.receptive_field)



# from mpunet import models
# dict_parameter = models.__dict__
# import pprint
#
# pprint.pprint(dict_parameter)
#
# model_names = 'unet'
# print(dict_parameter[model_names])
#
# sb = dict_parameter[model_names]()