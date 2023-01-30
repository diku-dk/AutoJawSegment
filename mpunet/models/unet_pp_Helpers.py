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


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    return x


def conv2d_block(filters: int):
    return Conv2D(filters=filters, kernel_size=(3, 3), padding='same', kernel_regularizer=l2(0.),
                  bias_regularizer=l2(0.))


def conv2d_block(self, kernel_reg, filters, l_name, in_):
    conv = Conv2D(int(filters * self.cf), self.kernel_size,
                  activation=self.activation, padding=self.padding,
                  kernel_regularizer=kernel_reg,
                  name=l_name + "_conv1")(in_)
    conv = Conv2D(int(filters * self.cf), self.kernel_size,
                  activation=self.activation, padding=self.padding,
                  kernel_regularizer=kernel_reg,
                  name=l_name + "_conv2")(conv)
    bn = BatchNormalization(name=l_name + "_BN")(conv)
    return bn

def conv2d_up_block_1(self, kernel_reg, filters, l_name, in_):
    up = UpSampling2D(size=(2, 2), name=l_name + "_up")(in_)
    conv = Conv2D(int(filters * self.cf), 2,
                  activation=self.activation,
                  padding=self.padding, kernel_regularizer=kernel_reg,
                  name=l_name + "_conv1")(up)
    bn = BatchNormalization(name=l_name + "_BN1")(conv)
    return bn

def conv2d_up_block_2(self, kernel_reg, filters, l_name, merge):
    conv = Conv2D(int(filters * self.cf), self.kernel_size,
                  activation=self.activation, padding=self.padding,
                  kernel_regularizer=kernel_reg,
                  name=l_name + "_conv2"
                  )(merge)
    conv = Conv2D(int(filters * self.cf), self.kernel_size,
                  activation=self.activation, padding=self.padding,
                  kernel_regularizer=kernel_reg,
                  name=l_name + "_conv3"
                  )(conv)
    in_ = BatchNormalization(
                            name=l_name + "_BN2"
                             )(conv)

    return in_

    return Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='same')


model_input = Input((480, 480, 1))
x00 = conv2d(filters=int(16 * number_of_filters))(model_input)
x00 = BatchNormalization()(x00)
x00 = LeakyReLU(0.01)(x00)
x00 = Dropout(0.2)(x00)
x00 = conv2d(filters=int(16 * number_of_filters))(x00)
x00 = BatchNormalization()(x00)
x00 = LeakyReLU(0.01)(x00)
x00 = Dropout(0.2)(x00)
p0 = MaxPooling2D(pool_size=(2, 2))(x00)

x10 = conv2d(filters=int(32 * number_of_filters))(p0)
x10 = BatchNormalization()(x10)
x10 = LeakyReLU(0.01)(x10)
x10 = Dropout(0.2)(x10)
x10 = conv2d(filters=int(32 * number_of_filters))(x10)
x10 = BatchNormalization()(x10)
x10 = LeakyReLU(0.01)(x10)
x10 = Dropout(0.2)(x10)
p1 = MaxPooling2D(pool_size=(2, 2))(x10)

x01 = conv2dtranspose(int(16 * number_of_filters))(x10)
x01 = concatenate([x00, x01])
x01 = conv2d(filters=int(16 * number_of_filters))(x01)
x01 = BatchNormalization()(x01)
x01 = LeakyReLU(0.01)(x01)
x01 = conv2d(filters=int(16 * number_of_filters))(x01)
x01 = BatchNormalization()(x01)
x01 = LeakyReLU(0.01)(x01)
x01 = Dropout(0.2)(x01)

x20 = conv2d(filters=int(64 * number_of_filters))(p1)
x20 = BatchNormalization()(x20)
x20 = LeakyReLU(0.01)(x20)
x20 = Dropout(0.2)(x20)
x20 = conv2d(filters=int(64 * number_of_filters))(x20)
x20 = BatchNormalization()(x20)
x20 = LeakyReLU(0.01)(x20)
x20 = Dropout(0.2)(x20)
p2 = MaxPooling2D(pool_size=(2, 2))(x20)

x11 = conv2dtranspose(int(16 * number_of_filters))(x20)
x11 = concatenate([x10, x11])
x11 = conv2d(filters=int(16 * number_of_filters))(x11)
x11 = BatchNormalization()(x11)
x11 = LeakyReLU(0.01)(x11)
x11 = conv2d(filters=int(16 * number_of_filters))(x11)
x11 = BatchNormalization()(x11)
x11 = LeakyReLU(0.01)(x11)
x11 = Dropout(0.2)(x11)

x02 = conv2dtranspose(int(16 * number_of_filters))(x11)
x02 = concatenate([x00, x01, x02])
x02 = conv2d(filters=int(16 * number_of_filters))(x02)
x02 = BatchNormalization()(x02)
x02 = LeakyReLU(0.01)(x02)
x02 = conv2d(filters=int(16 * number_of_filters))(x02)
x02 = BatchNormalization()(x02)
x02 = LeakyReLU(0.01)(x02)
x02 = Dropout(0.2)(x02)

x30 = conv2d(filters=int(128 * number_of_filters))(p2)
x30 = BatchNormalization()(x30)
x30 = LeakyReLU(0.01)(x30)
x30 = Dropout(0.2)(x30)
x30 = conv2d(filters=int(128 * number_of_filters))(x30)
x30 = BatchNormalization()(x30)
x30 = LeakyReLU(0.01)(x30)
x30 = Dropout(0.2)(x30)
p3 = MaxPooling2D(pool_size=(2, 2))(x30)

x21 = conv2dtranspose(int(16 * number_of_filters))(x30)
x21 = concatenate([x20, x21])
x21 = conv2d(filters=int(16 * number_of_filters))(x21)
x21 = BatchNormalization()(x21)
x21 = LeakyReLU(0.01)(x21)
x21 = conv2d(filters=int(16 * number_of_filters))(x21)
x21 = BatchNormalization()(x21)
x21 = LeakyReLU(0.01)(x21)
x21 = Dropout(0.2)(x21)

x12 = conv2dtranspose(int(16 * number_of_filters))(x21)
x12 = concatenate([x10, x11, x12])
x12 = conv2d(filters=int(16 * number_of_filters))(x12)
x12 = BatchNormalization()(x12)
x12 = LeakyReLU(0.01)(x12)
x12 = conv2d(filters=int(16 * number_of_filters))(x12)
x12 = BatchNormalization()(x12)
x12 = LeakyReLU(0.01)(x12)
x12 = Dropout(0.2)(x12)

x03 = conv2dtranspose(int(16 * number_of_filters))(x12)
x03 = concatenate([x00, x01, x02, x03])
x03 = conv2d(filters=int(16 * number_of_filters))(x03)
x03 = BatchNormalization()(x03)
x03 = LeakyReLU(0.01)(x03)
x03 = conv2d(filters=int(16 * number_of_filters))(x03)
x03 = BatchNormalization()(x03)
x03 = LeakyReLU(0.01)(x03)
x03 = Dropout(0.2)(x03)

m = conv2d(filters=int(256 * number_of_filters))(p3)
m = BatchNormalization()(m)
m = LeakyReLU(0.01)(m)
m = conv2d(filters=int(256 * number_of_filters))(m)
m = BatchNormalization()(m)
m = LeakyReLU(0.01)(m)
m = Dropout(0.2)(m)

x31 = conv2dtranspose(int(128 * number_of_filters))(m)
x31 = concatenate([x31, x30])
x31 = conv2d(filters=int(128 * number_of_filters))(x31)
x31 = BatchNormalization()(x31)
x31 = LeakyReLU(0.01)(x31)
x31 = conv2d(filters=int(128 * number_of_filters))(x31)
x31 = BatchNormalization()(x31)
x31 = LeakyReLU(0.01)(x31)
x31 = Dropout(0.2)(x31)

x22 = conv2dtranspose(int(64 * number_of_filters))(x31)
x22 = concatenate([x22, x20, x21])
x22 = conv2d(filters=int(64 * number_of_filters))(x22)
x22 = BatchNormalization()(x22)
x22 = LeakyReLU(0.01)(x22)
x22 = conv2d(filters=int(64 * number_of_filters))(x22)
x22 = BatchNormalization()(x22)
x22 = LeakyReLU(0.01)(x22)
x22 = Dropout(0.2)(x22)

x13 = conv2dtranspose(int(32 * number_of_filters))(x22)
x13 = concatenate([x13, x10, x11, x12])
x13 = conv2d(filters=int(32 * number_of_filters))(x13)
x13 = BatchNormalization()(x13)
x13 = LeakyReLU(0.01)(x13)
x13 = conv2d(filters=int(32 * number_of_filters))(x13)
x13 = BatchNormalization()(x13)
x13 = LeakyReLU(0.01)(x13)
x13 = Dropout(0.2)(x13)

x04 = conv2dtranspose(int(16 * number_of_filters))(x13)
x04 = concatenate([x04, x00, x01, x02, x03], axis=3)
x04 = conv2d(filters=int(16 * number_of_filters))(x04)
x04 = BatchNormalization()(x04)
x04 = LeakyReLU(0.01)(x04)
x04 = conv2d(filters=int(16 * number_of_filters))(x04)
x04 = BatchNormalization()(x04)
x04 = LeakyReLU(0.01)(x04)
x04 = Dropout(0.2)(x04)
output = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax')(x04)
model = tf.keras.Model(inputs=[model_input], outputs=[output])




def UEfficientNet(input_shape=(None, None, 3), dropout_rate=0.1):
    backbone = EfficientNetB4(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape)
    input = backbone.input
    start_neurons = 8

    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same", name='conv_middle')(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
    deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
    deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
    uconv4 = concatenate([deconv4, conv4])
    uconv4 = Dropout(dropout_rate)(uconv4)

    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    #     uconv4 = residual_block(uconv4,start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)  # conv1_2

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
    deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)
    conv3 = backbone.layers[154].output
    uconv3 = concatenate([deconv3, deconv4_up1, conv3])
    uconv3 = Dropout(dropout_rate)(uconv3)

    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    #     uconv3 = residual_block(uconv3,start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2)
    conv2 = backbone.layers[92].output
    uconv2 = concatenate([deconv2, deconv3_up1, deconv4_up2, conv2])

    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    #     uconv2 = residual_block(uconv2,start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[30].output
    uconv1 = concatenate([deconv1, deconv2_up1, deconv3_up2, deconv4_up3, conv1])

    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    #     uconv1 = residual_block(uconv1,start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    #     uconv0 = residual_block(uconv0,start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)

    uconv0 = Dropout(dropout_rate / 2)(uconv0)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv0)

    model = Model(input, output_layer)
    model.name = 'u-xception'

    return model


import tensorflow as tf
from tensorflow.keras.layers import concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Dropout, MaxPooling2D, Conv2DTranspose


batch_size = 2
epochs = 25
number_of_filters = 2
num_classes = 3
fil_coef = 2
leaky_alpha = 0.1
dropout_rate = 0.25


K.clear_session()
img_size = 256
model = UEfficientNet(input_shape=(img_size, img_size, 3), dropout_rate=0.5)