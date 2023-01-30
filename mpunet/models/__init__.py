from .unet import UNet
from .unet3D import UNet3D
from .fusion_model import FusionModel
from .model_init import model_initializer
from .multitask_unet2d import MultiTaskUNet2D

from .deepLabV3 import DeepLabV3Plus
from .unet2plus import UNet2Plus

from .unet_pp import UNet_PlusPlus

from .vnet import VNet
from .unet_2_path import UNet_2_path
from .vnet_double import VNetDouble

from .unet_double import UNetDouble
from.unetAttention import UNetAttention

# Prepare a dictionary mapping from model names to data prep. functions
from mpunet.preprocessing import data_preparation_funcs as dpf

PREPARATION_FUNCS = {
    "UNet": dpf.prepare_for_multi_view_unet,
    "UNet3D": dpf.prepare_for_3d_unet,
    "MultiTaskUNet2D": dpf.prepare_for_multi_task_2d,
    "DeepLabV3Plus": dpf.prepare_for_multi_view_unet,
    "UNet_PlusPlus": dpf.prepare_for_multi_view_unet,
    "VNet": dpf.prepare_for_multi_view_unet,
    "VNetDouble": dpf.prepare_for_multi_view_unet,
    "UNetDouble": dpf.prepare_for_multi_view_unet,
    "UNetVAE": dpf.prepare_for_multi_view_unet,
    "UNet2Plus": dpf.prepare_for_multi_view_unet,
    "UNetAttention": dpf.prepare_for_multi_view_unet,
}
