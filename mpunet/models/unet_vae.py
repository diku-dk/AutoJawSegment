from mpunet.models.unet import UNet

class UNetVAE(UNet):
    """
    2D UNet implementation with batch normalization and complexity factor adj.

    See original paper at http://arxiv.org/abs/1505.04597
    """

    def __init__(self, *args,  **kwargs):

        super().__init__(*args,  **kwargs)
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
        # Shows the number of pixels cropped of the input image to the output
        self.label_crop = np.array([[0, 0], [0, 0]])

        self.build_res = build_res

        self.init_filters = init_filters

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

            if self.build_res:
                in_ = Conv2D(int(filters * self.cf), 1,
                             activation=self.activation, padding=self.padding,
                             kernel_regularizer=kernel_reg,
                             name=l_name + "_conv1_res")(in_)
                conv = Add()([conv, in_])

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

        if self.build_res:
            in_ = Conv2D(int(filters * self.cf), 1,
                         activation=self.activation, padding=self.padding,
                         kernel_regularizer=kernel_reg,
                         name=name + "_conv1_res")(in_)
            conv = Add()([conv, in_])

        conv = BatchNormalization(name=name + "_BN")(conv)

        return conv

    def _create_upsample(self, in_, res_conns, filters, kernel_reg=None,
                         name="upsample"):
        residual_connections = res_conns[::-1]
        for i in range(self.depth):
            l_name = name + "_L%i" % i
            # Reduce filter count
            filters /= 2

            # want to use 3x3 here instead.
            up = UpSampling2D(size=(2, 2), name=l_name + "_up")(in_)
            conv = Conv2D(int(filters * self.cf), 2,
                          activation=self.activation,
                          padding=self.padding, kernel_regularizer=kernel_reg,
                          name=l_name + "_conv1")(up)

            bn = BatchNormalization(name=l_name + "_BN1")(conv)

            conv = Conv2D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv2")(merge)
            conv = Conv2D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kernel_reg,
                          name=l_name + "_conv3")(conv)

            if self.build_res:
                up = Conv2D(int(filters * self.cf), 1,
                             activation=self.activation, padding=self.padding,
                             kernel_regularizer=kernel_reg,
                             name=l_name + "_conv2_res")(up)
                conv = Add()([conv, up])

            in_ = BatchNormalization(name=l_name + "_BN2")(conv)

        return in_

    def init_model(self):
        """
        Build the UNet model with the specified input image shape.
        """
        inputs = Input(shape=self.img_shape)

        # if self.weight_map:
        #     input2 = Input(shape=self.img_shape)

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
        bn = self._create_bottom(in_, filters, kr)

        """
        Up-sampling
        """
        bn = self._create_upsample(bn, residual_cons, filters, kr)

        """
        Output modeling layer
        """
        # this is just to make sure that the name of final layer is different accross models
        final_layer_name = 'final_11_conv' + self.out_activation + str(np.random.randn())

        out = Conv2D(1, 1, activation='sigmoid', name=final_layer_name)(bn)

        if self.flatten_output:
            out = Reshape([self.img_shape[0] * self.img_shape[1],
                           self.n_classes], name='flatten_output')(out)


        return [inputs], [out]



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
