from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
                                    Concatenate, Conv2D, MaxPooling2D, \
                                    UpSampling2D, Reshape
import numpy as np

class UNet_PlusPlus(Model):
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
                 init_filters=64,
                 **kwargs):
        """
        n_classes (int):
            The number of classes to model, gives the number of filters in the
            final 1x1 conv layer.
        img_rows, img_cols (int, int):
            Image dimensions. Note that depending on image dims cropping may
            be necessary. To avoid this, use image dimensions DxD for which
            D * (1/2)^n is an integer, where n is the number of (2x2)
            max-pooling layers; in this implementation 4.
            For n=4, D \in {..., 192, 208, 224, 240, 256, ...} etc.
        dim (int):
            img_rows and img_cols will both be set to 'dim'
        n_channels (int):
            Number of channels in the input image.
        depth (int):
            Number of conv blocks in encoding layer (number of 2x2 max pools)
            Note: each block doubles the filter count while halving the spatial
            dimensions of the features.
        out_activation (string):
            Activation function of output 1x1 conv layer. Usually one of
            'softmax', 'sigmoid' or 'linear'.
        activation (string):
            Activation function for convolution layers
        kernel_size (int):
            Kernel size for convolution layers
        padding (string):
            Padding type ('same' or 'valid')
        complexity_factor (int/float):
            Use int(N * sqrt(complexity_factor)) number of filters in each
            2D convolution layer instead of default N.
        flatten_output (bool):
            Flatten the output to array of shape [batch_size, -1, n_classes]
        l2_reg (float in [0, 1])
            L2 regularization on Conv2D weights
        logger (mpunet.logging.Logger | ScreenLogger):
            MutliViewUNet.Logger object, logging to files or screen.
        """
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
        self.init_filters = init_filters

        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        names = [x.__class__.__name__ for x in self.layers]
        index = names.index("UpSampling2D")
        self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]

        # Log the model definition
        self.log()

    def _create_encoder(self, in_, init_filters, kernel_reg=None,
                        name="encoder"):
        filters = init_filters
        residual_connections = []

        for i in range(self.depth):
            l_name = name + "_L%i" % i
            conv = Conv2D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv1")(in_)
            conv = Conv2D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv2")(conv)
            bn = BatchNormalization(name=l_name + "_BN")(conv)

            in_ = MaxPooling2D(pool_size=(2, 2), name=l_name + "_pool")(bn)

            # Update filter count and add bn layer to list for residual conn.
            filters *= 2
            residual_connections.append(bn)
        return in_, residual_connections, filters

    def _create_bottom(self, in_, filters, kernel_reg=None, name="bottom"):
        conv = Conv2D(int(filters * self.cf), self.kernel_size,
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kernel_reg,
                      name=name + "_conv1")(in_)
        conv = Conv2D(int(filters * self.cf), self.kernel_size,
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kernel_reg,
                      name=name + "_conv2")(conv)
        bn = BatchNormalization(name=name + "_BN")(conv)

        return bn

    def _create_upsample(self, m, res_conns, init_filters, kernel_reg=None,
                         name="upsample"):
        # residual_connections = res_conns[::-1]

        filters = init_filters

        [x00, x10, x20, x30] = res_conns

        ############################## x , 1 path ##############################
        i = 0
        l_name = name + "_L%i" % i

        x01 = self.conv2d_up_block_1(kernel_reg, filters, l_name, x10)
        x00_cropped = self.crop_nodes_to_match(x00, x01)
        x01 = Concatenate(axis=-1,
                            #name=l_name + "_concat"
                          )([x00_cropped, x01])
        x01 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x01)

        i += 1
        filters *= 2

        x11 = self.conv2d_up_block_1( kernel_reg, filters, l_name, x20)
        x10_cropped = self.crop_nodes_to_match(x10, x11)
        x11 = Concatenate(axis=-1,
                            #name=l_name + "_concat"
                          )([x10_cropped, x11])
        x11 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x11)

        i += 1
        filters *= 2
        x21 = self.conv2d_up_block_1(kernel_reg, filters, l_name, x30)
        x20_cropped = self.crop_nodes_to_match(x20, x21)
        x21 = Concatenate(axis=-1,
                            #name=l_name + "_concat"
                          )([x20_cropped, x21])
        x21 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x21)

        i += 1
        filters *= 2
        x31 = self.conv2d_up_block_1(kernel_reg, filters, l_name, m)
        x30_cropped = self.crop_nodes_to_match(x30, x31)
        x31 = Concatenate(axis=-1,
                            # name=l_name + "_concat"
                          )([x30_cropped, x31])
        x31 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x31)

        ############################## x , 2 path ##############################
        filters = init_filters
        i = 0
        x02 = self.conv2d_up_block_1(kernel_reg, filters, l_name, x11)
        x00_cropped = self.crop_nodes_to_match(x00, x02)
        x01_cropped = self.crop_nodes_to_match(x01, x02)
        x02 = Concatenate(axis=-1,
                            #name=l_name + "_concat"
                          )([x00_cropped, x01_cropped, x02])
        x02 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x02)

        filters *= 2
        i += 1
        x12 = self.conv2d_up_block_1(kernel_reg, filters, l_name, x21)
        x10_cropped = self.crop_nodes_to_match(x10, x12)
        x11_cropped = self.crop_nodes_to_match(x11, x12)
        x12 = Concatenate(axis=-1,
                            #name=l_name + "_concat"
                          )([x10_cropped, x11_cropped, x12])
        x12 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x12)

        filters *= 2
        i += 1
        x22 = self.conv2d_up_block_1(kernel_reg, filters, l_name, x31)
        x20_cropped = self.crop_nodes_to_match(x20, x22)
        x21_cropped = self.crop_nodes_to_match(x21, x22)
        x22 = Concatenate(axis=-1,
                            # name=l_name + "_concat"
                          )([x20_cropped, x21_cropped, x22])
        x22 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x22)

        ############################## x , 3 path ##############################
        filters = init_filters
        i = 0
        x03 = self.conv2d_up_block_1(kernel_reg, filters, l_name, x12)
        x00_cropped = self.crop_nodes_to_match(x00, x03)
        x01_cropped = self.crop_nodes_to_match(x01, x03)
        x02_cropped = self.crop_nodes_to_match(x02, x03)
        x03 = Concatenate(axis=-1,
                            # name=l_name + "_concat"
                          )([x00_cropped, x01_cropped,
                                                      x02_cropped, x03])
        x03 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x03)

        filters *= 2
        i += 1

        x13 = self.conv2d_up_block_1(kernel_reg, filters, l_name, x22)
        x10_cropped = self.crop_nodes_to_match(x10, x13)
        x11_cropped = self.crop_nodes_to_match(x11, x13)
        x12_cropped = self.crop_nodes_to_match(x12, x13)

        x13 = Concatenate(axis=-1,
                            # name=l_name + "_concat"
                          )([x10_cropped, x11_cropped,
                                                      x12_cropped, x13])
        x13 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x13)


        ############################## x , 4 path ##############################

        filters = init_filters
        i = 0

        x04 = self.conv2d_up_block_1(kernel_reg, filters, l_name, x13)
        x00_cropped = self.crop_nodes_to_match(x00, x04)
        x01_cropped = self.crop_nodes_to_match(x01, x04)
        x02_cropped = self.crop_nodes_to_match(x02, x04)
        x03_cropped = self.crop_nodes_to_match(x03, x04)

        x04 = Concatenate(axis=-1,
                            # name=l_name + "_concat"
                          )([x00_cropped, x01_cropped,
                                                      x02_cropped, x03_cropped,
                                                      x04])
        x04 = self.conv2d_up_block_2(kernel_reg, filters, l_name, x04)

        return x04

    def init_model(self):
        """
        Build the UNet model with the specified input image shape.
        """
        inputs = Input(shape=self.img_shape)

        # Apply regularization if not None or 0
        kr = regularizers.l2(self.l2_reg) if self.l2_reg else None

        """
        Encoding path
        """
        in_, residual_cons, filters = self._create_encoder(in_=inputs,
                                                           init_filters=self.init_filters,
                                                           kernel_reg=kr)
        """
        Bottom (no max-pool)
        """
        m = self._create_bottom(in_, filters, kr)

        """
        Up-sampling
        """
        bn = self._create_upsample(m, residual_cons, init_filters=self.init_filters,
                                   kernel_reg=kr, name="upsample")

        """
        Output modeling layer
        """
        # this is just to make sure that the name of final layer is different accross models
        final_layer_name = 'final_11_conv' + str(np.random.randn())
        if self.n_classes == 1:
            out = Conv2D(self.n_classes, 1, activation='sigmoid', name=final_layer_name)(bn)
        else:
            out = Conv2D(self.n_classes, 1, activation=self.out_activation, name=final_layer_name)(bn)
        if self.flatten_output:
            if self.n_classes == 1:
                out = Reshape([self.img_shape[0] * self.img_shape[1]],
                               name='flatten_output')(out)
            else:
                out = Reshape([self.img_shape[0]*self.img_shape[1],
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
            cr = np.array([c//2, c//2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping2D(cr)(node1)
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1

    def conv2d_up_block_1(self, kernel_reg, filters, l_name, in_):
        up = UpSampling2D(size=(2, 2),
                          # name=l_name + "_up"
                          )(in_)
        conv = Conv2D(int(filters * self.cf), 2,
                      activation=self.activation,
                      padding=self.padding, kernel_regularizer=kernel_reg,
                      # name=l_name + "_conv1"
                      )(up)
        bn = BatchNormalization(
                            # name=l_name + "_BN1"
                                )(conv)
        return bn

    def conv2d_up_block_2(self, kernel_reg, filters, l_name, merge):
        conv = Conv2D(int(filters * self.cf), self.kernel_size,
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kernel_reg,
                      #name=l_name + "_conv2"
                      )(merge)
        conv = Conv2D(int(filters * self.cf), self.kernel_size,
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kernel_reg,
                      #name=l_name + "_conv3"
                      )(conv)
        in_ = BatchNormalization(
            # name=l_name + "_BN2"
        )(conv)

        return in_

    def log(self):
        self.logger("UNet Model Summary\n------------------")
        self.logger("Image rows:        %i" % self.img_shape[0])
        self.logger("Image cols:        %i" % self.img_shape[1])
        self.logger("Image channels:    %i" % self.img_shape[2])
        self.logger("N classes:         %i" % self.n_classes)
        self.logger("CF factor:         %.3f" % self.cf**2)
        self.logger("Depth:             %i" % self.depth)
        self.logger("l2 reg:            %s" % self.l2_reg)
        self.logger("Padding:           %s" % self.padding)
        self.logger("Conv activation:   %s" % self.activation)
        self.logger("Out activation:    %s" % self.out_activation)
        self.logger("Receptive field:   %s" % self.receptive_field)
        self.logger("N params:          %i" % self.count_params())
        self.logger("Output:            %s" % self.output)
        self.logger("Crop:              %s" % (self.label_crop if np.sum(self.label_crop) != 0 else "None"))

