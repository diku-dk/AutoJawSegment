
from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D, Reshape
import numpy as np
from mpunet.models.deepLabHelpers import *
from mpunet.models.deepLabHelpers import _conv2d_same, _xception_block, _make_divisible, _inverted_res_block


class DeepLabV3Plus(Model):
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
                 weight_pretrained='pascal_voc',
                 input_tensor=None,
                 backbone='xception',
                 OS=16,
                 alpha=1.,
                 **kwargs):
        super().__init__()
        if not ((img_rows and img_cols) or dim):
            raise ValueError("Must specify either img_rows and img_col or dim")
        if dim:
            img_rows, img_cols = dim, dim

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
        self.weight_pretrained = weight_pretrained
        self.input_tensor = input_tensor
        self.backbone = backbone
        self.alpha = alpha
        self.OS = OS

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        names = [x.__class__.__name__ for x in self.layers]

        # index = names.index("UpSampling2D")
        # self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]

        # Log the model definition
        self.log()

    def init_model(self):

        if not (self.weight_pretrained in {'pascal_voc', 'cityscapes', None}):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization), `pascal_voc`, or `cityscapes` '
                             '(pre-trained on PASCAL VOC)')

        if not (self.backbone in {'xception', 'mobilenetv2'}):
            raise ValueError('The `backbone` argument should be either '
                             '`xception`  or `mobilenetv2` ')

        if self.input_tensor is None:
            img_input = Input(shape=self.img_shape)
        else:
            img_input = self.input_tensor

        if self.backbone == 'xception':
            if self.OS == 8:
                entry_block3_stride = 1
                middle_block_rate = 2  # ! Not mentioned in paper, but required
                exit_block_rates = (2, 4)
                atrous_rates = (12, 24, 36)
            else:
                entry_block3_stride = 2
                middle_block_rate = 1
                exit_block_rates = (1, 2)
                atrous_rates = (6, 12, 18)

            first_layer_name = 'entry_flow_conv1_1'
            # if self.img_shape[-1] !=3:
            #     first_layer_name += '_custom'
            x = Conv2D(32, (3, 3), strides=(2, 2),
                       name=first_layer_name, use_bias=False, padding='same')(img_input)
            x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
            x = Activation(tf.nn.relu)(x)
            x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
            x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
            x = Activation(tf.nn.relu)(x)

            x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                                skip_connection_type='conv', stride=2,
                                depth_activation=False)
            x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                       skip_connection_type='conv', stride=2,
                                       depth_activation=False, return_skip=True)

            x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                                skip_connection_type='conv', stride=entry_block3_stride,
                                depth_activation=False)
            for i in range(16):
                x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                    skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                    depth_activation=False)

            x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                                skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                                depth_activation=False)
            x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                                skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                                depth_activation=True)

        else:
            OS = 8
            first_block_filters = _make_divisible(32 * self.alpha, 8)
            x = Conv2D(first_block_filters,
                       kernel_size=3,
                       strides=(2, 2), padding='same',
                       use_bias=False, name='Conv')(img_input)
            x = BatchNormalization(
                epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
            x = Activation(tf.nn.relu6, name='Conv_Relu6')(x)

            x = _inverted_res_block(x, filters=16, alpha=self.alpha, stride=1,
                                    expansion=1, block_id=0, skip_connection=False)

            x = _inverted_res_block(x, filters=24, alpha=self.alpha, stride=2,
                                    expansion=6, block_id=1, skip_connection=False)
            x = _inverted_res_block(x, filters=24, alpha=self.alpha, stride=1,
                                    expansion=6, block_id=2, skip_connection=True)

            x = _inverted_res_block(x, filters=32, alpha=self.alpha, stride=2,
                                    expansion=6, block_id=3, skip_connection=False)
            x = _inverted_res_block(x, filters=32, alpha=self.alpha, stride=1,
                                    expansion=6, block_id=4, skip_connection=True)
            x = _inverted_res_block(x, filters=32, alpha=self.alpha, stride=1,
                                    expansion=6, block_id=5, skip_connection=True)

            # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
            x = _inverted_res_block(x, filters=64, alpha=self.alpha, stride=1,  # 1!
                                    expansion=6, block_id=6, skip_connection=False)
            x = _inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=7, skip_connection=True)
            x = _inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=8, skip_connection=True)
            x = _inverted_res_block(x, filters=64, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=9, skip_connection=True)

            x = _inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=10, skip_connection=False)
            x = _inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=11, skip_connection=True)
            x = _inverted_res_block(x, filters=96, alpha=self.alpha, stride=1, rate=2,
                                    expansion=6, block_id=12, skip_connection=True)

            x = _inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=2,  # 1!
                                    expansion=6, block_id=13, skip_connection=False)
            x = _inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=4,
                                    expansion=6, block_id=14, skip_connection=True)
            x = _inverted_res_block(x, filters=160, alpha=self.alpha, stride=1, rate=4,
                                    expansion=6, block_id=15, skip_connection=True)

            x = _inverted_res_block(x, filters=320, alpha=self.alpha, stride=1, rate=4,
                                    expansion=6, block_id=16, skip_connection=False)

        # end of feature extractor

        # branching for Atrous Spatial Pyramid Pooling

        # Image Feature branch
        shape_before = tf.shape(x)
        b4 = GlobalAveragePooling2D()(x)
        # from (b_size, channels)->(b_size, 1, 1, channels)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
        b4 = Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation(tf.nn.relu)(b4)
        # upsample. have to use compat because of the option align_corners
        size_before = tf.keras.backend.int_shape(x)
        b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
                                                        method='bilinear', align_corners=True))(b4)
        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation(tf.nn.relu, name='aspp0_activation')(b0)

        # there are only 2 branches in mobilenetV2. not sure why
        if self.backbone == 'xception':
            # rate = 6 (12)
            b1 = SepConv_BN(x, 256, 'aspp1',
                            rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
            # rate = 12 (24)
            b2 = SepConv_BN(x, 256, 'aspp2',
                            rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
            # rate = 18 (36)
            b3 = SepConv_BN(x, 256, 'aspp3',
                            rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

            # concatenate ASPP branches & project
            x = Concatenate()([b4, b0, b1, b2, b3])
        else:
            x = Concatenate()([b4, b0])

        x = Conv2D(256, (1, 1), padding='same',
                   use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation(tf.nn.relu)(x)
        x = Dropout(0.1)(x)
        # DeepLab v.3+ decoder

        if self.backbone == 'xception':
            # Feature projection
            # x4 (x2) block
            skip_size = tf.keras.backend.int_shape(skip1)
            x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                            skip_size[1:3],
                                                            method='bilinear', align_corners=True))(x)

            dec_skip1 = Conv2D(48, (1, 1), padding='same',
                               use_bias=False, name='feature_projection0')(skip1)
            dec_skip1 = BatchNormalization(
                name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
            dec_skip1 = Activation(tf.nn.relu)(dec_skip1)
            x = Concatenate()([x, dec_skip1])
            x = SepConv_BN(x, 256, 'decoder_conv0',
                           depth_activation=True, epsilon=1e-5)
            x = SepConv_BN(x, 256, 'decoder_conv1',
                           depth_activation=True, epsilon=1e-5)

        # you can use it with arbitary number of classes
        if (self.weight_pretrained == 'pascal_voc' and self.n_classes == 21) or (
                self.weight_pretrained == 'cityscapes' and self.n_classes == 19):
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        last_layer_name = last_layer_name + str(np.random.randn())

        x = Conv2D(self.n_classes, (1, 1), padding='same', name=last_layer_name)(x)

        size_before3 = tf.keras.backend.int_shape(img_input)
        x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
                                                        size_before3[1:3],
                                                        method='bilinear',
                                                        align_corners=True))(x)

        # # Ensure that the model takes into account
        # # any potential predecessors of `input_tensor`.
        # if self.input_tensor is not None:
        #     inputs = get_source_inputs(self.input_tensor)
        # else:
        #     inputs = img_input

        if self.n_classes == 1:
            x = Activation('sigmoid')(x)
        else:
            x = Activation('softmax')(x)

        if self.flatten_output:
            if self.n_classes == 1:
                x = Reshape([self.img_shape[0] * self.img_shape[1]],
                               name='flatten_output')(x)
            else:
                x = Reshape([self.img_shape[0]*self.img_shape[1],
                               self.n_classes], name='flatten_output')(x)

        return [img_input], [x]

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-1]
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            c = (s1 - s2).astype(np.int)
            cr = np.array([c//2, c//2]).T
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
        # self.logger("CF factor:         %.3f" % self.cf**2)
        # self.logger("Depth:             %i" % self.depth)
        # self.logger("l2 reg:            %s" % self.l2_reg)
        # self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        # self.logger("Out activation:    %s" % self.out_activation)
        # self.logger("Receptive field:   %s" % self.receptive_field)
        # self.logger("N params:          %i" % self.count_params())
        # self.logger("Output:            %s" % self.output)
        # self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))
