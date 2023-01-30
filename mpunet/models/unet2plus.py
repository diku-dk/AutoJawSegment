from tensorflow.keras import Sequential,Model
from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D, Reshape, ZeroPadding2D,\
                                    Dense, Conv2DTranspose, Activation
                                    
import numpy as np
from tensorflow import keras
import tensorflow as tf
from mpunet.models.utils import *

## Nested block
class UNet2Plus(Model):
    """
    2D UNet implementation with batch normalization and complexity factor adj.

    See original paper at http://arxiv.org/abs/1505.04597
    """
    def __init__(self,
                 n_classes,
                 img_rows=None,
                 img_cols=None,
                 dim=None,
                 n_channels=1,
                 depth=4,
                 out_activation="softmax",
                 activation="relu",
                 kernel_size=3,
                 padding="same",
                 complexity_factor=1,
                 flatten_output=False,
                 l2_reg=None,
                 logger=None,
                 **kwargs):
        super().__init__()
        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim

        # Set various attributes
        self.img_shape = (img_rows, img_cols, n_channels)
        self.num_class = n_classes
        self.cf = np.sqrt(complexity_factor)
        self.kernel_size = kernel_size
        self.activation = activation
        self.out_activation = out_activation
        self.l2_reg = l2_reg
        self.padding = padding
        self.depth = depth
        self.flatten_output = flatten_output
        self.img_rows=img_rows
        self.img_cols=img_cols
        self.kernel_reg = regularizers.l2(l2_reg) if l2_reg else None
        self.pool = MaxPooling2D(pool_size=(2, 2))
        self.Up = UpSampling2D(size=(2, 2))
	

        # Shows the number of pixels cropped of the input image to the output
        #self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        #names = [x.__class__.__name__ for x in self.layers]
        #index = names.index("UpSampling2D")
        #self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]

        # Log the model definition
        self.log()

    def conv_block_nested(self, inputs, mid_ch, out_ch ):
        x = Conv2D(mid_ch, kernel_size=3, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(out_ch, kernel_size=3, padding='same')(x)
        x = BatchNormalization()(x)
        output = Activation('relu')(x)
        return output

    def init_model(self):
        inputs = Input(shape=self.img_shape)
        num_class = self.num_class
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        x0_0 = self.conv_block_nested(inputs, filters[0], filters[0])
        x1_0 = self.conv_block_nested(self.pool(x0_0),filters[1], filters[1])
        x0_1 = self.conv_block_nested(Concatenate(axis=-1)([x0_0, self.Up(x1_0)]),filters[0], filters[0])

        x2_0 = self.conv_block_nested(self.pool(x1_0),filters[2], filters[2])
        x1_1 = self.conv_block_nested(Concatenate(axis=-1)([x1_0, self.Up(x2_0)]), filters[1], filters[1])
        x0_2 = self.conv_block_nested(Concatenate(axis=-1)([x0_0, x0_1, self.Up(x1_1)]),filters[0], filters[0])

        x3_0 = self.conv_block_nested(self.pool(x2_0),filters[3], filters[3])

        # x3_0 = DoubleAttention(input_channel=filters[3], name='DoubleAttention_1')(x3_0)

        x2_1 = self.conv_block_nested(Concatenate(axis=-1)([x2_0, self.Up(x3_0)]),filters[2], filters[2])
        x1_2 = self.conv_block_nested(Concatenate(axis=-1)([x1_0, x1_1, self.Up(x2_1)]),filters[1], filters[1])
        x0_3 = self.conv_block_nested(Concatenate(axis=-1)([x0_0, x0_1, x0_2, self.Up(x1_2)]),filters[0], filters[0])

        x = self.conv_block_nested(self.pool(x3_0),filters[4], filters[4])
        x = self.conv_block_nested(Concatenate(axis=-1)([x3_0, self.Up(x)]),filters[3], filters[3])
        x = self.conv_block_nested(Concatenate(axis=-1)([x2_0, x2_1, self.Up(x)]),filters[2], filters[2])
        x = self.conv_block_nested(Concatenate(axis=-1)([x1_0, x1_1, x1_2, self.Up(x)]),filters[1], filters[1])
        x = self.conv_block_nested(Concatenate(axis=-1)([x0_0, x0_1, x0_2, x0_3, self.Up(x)]),filters[0], filters[0])

        x = Conv2D(self.num_class, kernel_size=1, activation='softmax', name='final_layer'+self.out_activation+str(np.random.randn()))(x)
        if self.flatten_output:
            x = Reshape([self.img_shape[0]*self.img_shape[1],
                           self.num_class], name='flatten_output')(x)
        print('shapes of input and out are', inputs.shape, x.shape)
        return [inputs], [x]


    def log(self):
        self.logger("UNet Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.num_class)
        #self.logger("CF factor:         %.3f" % self.cf**2)
        #self.logger("Depth:             %i" % self.depth)
        #self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % self.out_activation)
        #self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        #self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))
