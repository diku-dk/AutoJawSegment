"""
Mathias Perslev & Peidi Xu
University of Copenhagen
"""

from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
    Concatenate, Conv2D, MaxPooling2D, \
    UpSampling2D, Reshape, Add, Conv2D, PReLU, ReLU, \
    Conv2DTranspose, add, concatenate, Input, Dropout, BatchNormalization, Activation
import numpy as np

from mpunet.models.utils import *

class VNetDouble(Model):

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
                 build_res=False,
                 init_filters=64,
                 weight_map=False,
                 **kwargs):

        super().__init__()
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim

        assert kernel_size in (3, 5)

        # Set logger or standard print wrapper
        self.logger = logger or ScreenLogger()

        # Set various attributes
        self.img_shape = (img_rows, img_cols, n_channels)
        self.n_classes = n_classes
        self.cf = np.sqrt(complexity_factor)
        self.kernel_size = kernel_size
        self.activation = activation
        self.out_activation = out_activation
        self.l2_reg = l2_reg
        self.padding = padding
        self.depth = depth
        self.flatten_output = flatten_output
        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        self.build_res = build_res

        self.init_filters = init_filters

        self.weight_map = weight_map

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        names = [x.__class__.__name__ for x in self.layers]
        index = names.index("Conv2DTranspose")
        self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]

        # Log the model definition
        self.log()

    def resBlock(self, conv, stage, keep_prob, stage_num=5, name="encoder"):  # 收缩路径

        inputs = conv

        name = f'stage_{stage}_{name}'

        for i in range(3 if stage >= 3 else stage+1):
            l_name = f'{name}_L_{i}'

            if self.kernel_size == 5:
                conv = Conv2D(self.init_filters * (2 ** (stage)), 5,
                              activation=None, padding='same', kernel_initializer='he_normal',
                              name=l_name + '_conv1')(conv)
            else:
                conv = Conv2D(self.init_filters * (2 ** (stage)), 3,
                              activation=None, padding='same', kernel_initializer='he_normal',
                              name=l_name + '_conv1')(conv)

                conv = BatchNormalization(name=l_name + "_BN_1")(conv)
                conv = ReLU()(conv)

                conv = Conv2D(self.init_filters * (2 ** (stage)), 3,
                              activation=None, padding='same', kernel_initializer='he_normal',
                              name=l_name + '_conv2')(conv)

            conv = BatchNormalization(name=l_name + "_BN")(conv)
            conv = ReLU()(conv)

        conv = DoubleAttention(input_channel=self.init_filters * (2 ** (stage)),
                                  name=l_name +'DoubleAttention')(conv)

        inputs = DoubleAttention(input_channel=self.init_filters * (2 ** (stage)),
                                    name=l_name + 'DoubleAttentionInput')(inputs)


        conv_add = ReLU()(add([inputs, conv]))

        if stage < stage_num:

            conv_downsample = Conv2D(self.init_filters * (2 ** (stage+1)), 2, strides=(2, 2), activation=None,
                                     padding='same', kernel_initializer='he_normal',
                                     name=name + '_conv_downsample')(conv_add)
            conv_downsample = BatchNormalization(name=name + "downsample_BN")(conv_downsample)
            conv_downsample = ReLU()(conv_downsample)

            return conv_downsample, conv_add  # 返回每个stage下采样后的结果,以及在相加之前的结果
        else:
            return conv, conv  # 返回相加之后的结果，为了和上面输出保持一致，所以重复输出

    def up_resBlock(self, forward_conv, input_conv, stage, name="upsample"):  # 扩展路径

        name = f'stage_{stage}_{name}'

        conv = concatenate([forward_conv, input_conv], axis=-1)
        # print('conv_concatenate:', conv.get_shape().as_list())
        for i in range(3 if stage >= 3 else stage):
            l_name = f'{name}_L_{i}'

            if self.kernel_size == 5:
                conv = Conv2D(self.init_filters * (2 ** (stage)), 5, activation=None,
                              padding='same', kernel_initializer='he_normal',
                              name=l_name + '_conv1')(conv)
            else:
                conv = Conv2D(self.init_filters * (2 ** (stage)), 3, activation=None,
                              padding='same', kernel_initializer='he_normal',
                              name=l_name + '_conv1')(conv)

                conv = BatchNormalization(name=l_name + "_BN_1")(conv)
                conv = ReLU()(conv)

                conv = Conv2D(self.init_filters * (2 ** (stage)), 3, activation=None,
                              padding='same', kernel_initializer='he_normal',
                              name=l_name + '_conv2')(conv)

            conv = BatchNormalization(name=l_name + "_BN")(conv)
            conv = ReLU()(conv)


            conv = DoubleAttention(input_channel=self.init_filters * (2 ** (stage)),
                                      name=l_name +'DoubleAttention')(conv)

            # print('conv_up_stage_%d:' % stage, conv.get_shape().as_list())  # 输出扩展路径中每个stage内的卷积
        if stage > 0:

            input_conv = DoubleAttention(input_channel=self.init_filters * (2 ** (stage)),
                                        name=l_name + 'DoubleAttentionInput')(input_conv)

            conv_add = ReLU()(add([input_conv, conv]))
            conv_upsample = Conv2DTranspose(self.init_filters * (2 ** (stage - 1)), 2, strides=(2, 2),
                                            padding='valid', activation=None, kernel_initializer='he_normal',
                                            name=name + '_conv_upsample'
                                            )(conv_add)
            conv_upsample = BatchNormalization(name=name + "upsample_BN")(conv_upsample)
            conv_upsample = ReLU()(conv_upsample)

            return conv_upsample
        else:
            return conv

    def init_model(self):
        keep_prob = 1.0
        features = []

        """
        Build the UNet model with the specified input image shape.
        """
        inputs = Input(shape=self.img_shape)

        if self.kernel_size == 5:

            x = Conv2D(self.init_filters, 5, activation=None,
                       padding='same', kernel_initializer='he_normal',
                       name="first_conv"
                       )(inputs)

        else:
            x = Conv2D(self.init_filters, 3, activation=None,
                       padding='same', kernel_initializer='he_normal',
                       name="first_conv_1"
                       )(inputs)
            x = BatchNormalization(name="first_BN_1")(x)
            x = ReLU()(x)

            x = Conv2D(self.init_filters, 3, activation=None,
                       padding='same', kernel_initializer='he_normal',
                       name="first_conv_2"
                       )(x)

        x = BatchNormalization(name="first_BN_2")(x)
        x = ReLU()(x)


        x = DoubleAttention(input_channel=self.init_filters,
                                  name="first_conv" + 'DoubleAttention')(x)



        for s in range(self.depth+1):
            x, feature = self.resBlock(x, s, keep_prob, self.depth)  # 调用收缩路径
            features.append(feature)

        #x = self._create_bottom(filters=self.init_filters * (2 ** self.depth),
        #                             in_=x,
        #                             name="bottom")


        conv_up = Conv2DTranspose(self.init_filters * (2 ** (self.depth - 1)), 2, strides=(2, 2),
                                  padding='valid', activation=None, kernel_initializer='he_normal',
                                  name="first_up_transpose"
                                  )(x)

        conv_up = BatchNormalization(name="first_up_bn")(conv_up)
        conv_up = ReLU()(conv_up)

        for d in range(self.depth)[::-1]:
            conv_up = self.up_resBlock(forward_conv=features[d], input_conv=conv_up, stage=d)  # 调用扩展路径
        # conv_up_surf = self.up_resBlock(features[d - 1], conv_up, d, name='upsample_surf')
        # out_surface = Conv2D(self.n_classes, 1, activation='sigmoid', name=final_layer_name+'surf')(bn_surface)
        # out_surface = Reshape([self.img_shape[0] * self.img_shape[1]],
        #                       name='flatten_surf')(out_surface)

        """
        Output modeling layer
        """
        # this is just to make sure that the name of final layer is different accross models
        final_layer_name = 'final_11_conv' + self.out_activation + str(np.random.randn())

        if self.n_classes == 1:
            out = Conv2D(self.n_classes, 1, activation='sigmoid', name=final_layer_name)(conv_up)
        else:
            out = Conv2D(self.n_classes, 1, activation=self.out_activation, name=final_layer_name)(conv_up)

        if self.flatten_output:
            if self.n_classes == 1:
                out = Reshape([self.img_shape[0] * self.img_shape[1]],
                              name='flatten_output')(out)
            else:
                out = Reshape([self.img_shape[0] * self.img_shape[1],
                               self.n_classes], name='flatten_output')(out)

        return [inputs], [out]

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-1]
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            c = (s1 - s2).astype(np.int)
            cr = np.array([c // 2, c // 2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping2D(cr)(node1)
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1

    def log(self):
        self.logger("UNet Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.n_classes)
        self.logger("CF factor:         %.3f" % self.cf ** 2)
        self.logger("Depth:             %i" % self.depth)
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % self.out_activation)
        self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))
