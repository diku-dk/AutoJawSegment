from tensorflow.keras.layers import GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, \
    Conv2D, Add, Activation, Lambda,Conv1D, Multiply
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
    Concatenate, Conv2D, MaxPooling2D, \
    UpSampling2D, Reshape, Add, Conv2D, PReLU, ReLU, \
    Conv2DTranspose, add, concatenate, Input, Dropout, BatchNormalization, Activation
import numpy as np
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K


class DropBlock1D(tensorflow.keras.layers.Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock1D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = tensorflow.keras.engine.base_layer.InputSpec(ndim=3)
        self.supports_masking = True

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock1D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self, feature_dim):
        """Get the number of activation units to drop"""
        feature_dim = K.cast(feature_dim, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / block_size) * (feature_dim / (feature_dim - block_size + 1.0))

    def _compute_valid_seed_region(self, seq_length):
        positions = K.arange(seq_length)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions >= half_block_size,
                        positions < seq_length - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            K.ones((seq_length,)),
            K.zeros((seq_length,)),
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        seq_length = shape[1]
        mask = K.random_binomial(shape, p=self._get_gamma(seq_length))
        mask *= self._compute_valid_seed_region(seq_length)
        mask = tensorflow.keras.layers.MaxPool1D(
            pool_size=self.block_size,
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask *\
                (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 1])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


class DropBlock2D(tensorflow.keras.layers.Layer):
    """See: https://arxiv.org/pdf/1810.12890.pdf"""

    def __init__(self,
                 block_size,
                 keep_prob,
                 sync_channels=False,
                 data_format=None,
                 **kwargs):
        """Initialize the layer.
        :param block_size: Size for each mask block.
        :param keep_prob: Probability of keeping the original feature.
        :param sync_channels: Whether to use the same dropout for all channels.
        :param data_format: 'channels_first' or 'channels_last' (default).
        :param kwargs: Arguments for parent class.
        """
        super(DropBlock2D, self).__init__(**kwargs)
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        self.data_format = K.normalize_data_format(data_format)
        self.input_spec = tensorflow.keras.engine.base_layer.InputSpec(ndim=4)
        self.supports_masking = True

    def get_config(self):
        config = {'block_size': self.block_size,
                  'keep_prob': self.keep_prob,
                  'sync_channels': self.sync_channels,
                  'data_format': self.data_format}
        base_config = super(DropBlock2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask=None):
        return mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def _get_gamma(self, height, width):
        """Get the number of activation units to drop"""
        height, width = K.cast(height, K.floatx()), K.cast(width, K.floatx())
        block_size = K.constant(self.block_size, dtype=K.floatx())
        return ((1.0 - self.keep_prob) / (block_size ** 2)) *\
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))

    def _compute_valid_seed_region(self, height, width):
        positions = K.concatenate([
            K.expand_dims(K.tile(K.expand_dims(K.arange(height), axis=1), [1, width]), axis=-1),
            K.expand_dims(K.tile(K.expand_dims(K.arange(width), axis=0), [height, 1]), axis=-1),
        ], axis=-1)
        half_block_size = self.block_size // 2
        valid_seed_region = K.switch(
            K.all(
                K.stack(
                    [
                        positions[:, :, 0] >= half_block_size,
                        positions[:, :, 1] >= half_block_size,
                        positions[:, :, 0] < height - half_block_size,
                        positions[:, :, 1] < width - half_block_size,
                    ],
                    axis=-1,
                ),
                axis=-1,
            ),
            K.ones((height, width)),
            K.zeros((height, width)),
        )
        return K.expand_dims(K.expand_dims(valid_seed_region, axis=0), axis=-1)

    def _compute_drop_mask(self, shape):
        height, width = shape[1], shape[2]
        mask = K.random_binomial(shape, p=self._get_gamma(height, width))
        mask *= self._compute_valid_seed_region(height, width)
        mask = tensorflow.keras.layers.MaxPool2D(
            pool_size=(self.block_size, self.block_size),
            padding='same',
            strides=1,
            data_format='channels_last',
        )(mask)
        return 1.0 - mask

    def call(self, inputs, training=None):

        def dropped_inputs():
            outputs = inputs
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 2, 3, 1])
            shape = K.shape(outputs)
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], shape[1], shape[2], 1])
            else:
                mask = self._compute_drop_mask(shape)
            outputs = outputs * mask *\
                (K.cast(K.prod(shape), dtype=K.floatx()) / K.sum(mask))
            if self.data_format == 'channels_first':
                outputs = K.permute_dimensions(outputs, [0, 3, 1, 2])
            return outputs

        return K.in_train_phase(dropped_inputs, inputs, training=training)


def spatial_attention(input_feature):
    kernel_size = 7

    if K.image_data_format() == "channels_first":
        channel = input_feature._keras_shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature._keras_shape[-1]
        cbam_feature = input_feature

    # avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    # assert avg_pool._keras_shape[-1] == 1
    # max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    # assert max_pool._keras_shape[-1] == 1

    avg_pool = tf.math.reduce_mean(cbam_feature, axis=-1, keepdims=True)
    assert avg_pool.get_shape()[-1] == 1
    max_pool = tf.math.reduce_sum(cbam_feature, axis=-1, keepdims=True)
    assert max_pool.get_shape()[-1] == 1

    concat = Concatenate(axis=3)([avg_pool, max_pool])

    # assert concat._keras_shape[-1] == 2

    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)

    # assert cbam_feature._keras_shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])


class SpatialAttention(tensorflow.keras.layers.Layer):
    def __init__(self, name='SpatialAttention', kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(name=name, **kwargs)

        self.conv2D = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)
        self.multiply = Multiply()
        self.concat = Concatenate()


    def call(self, x):
        avg_pool = tf.math.reduce_mean(x, axis=-1, keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.math.reduce_sum(x, axis=-1, keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        concat = self.concat([avg_pool, max_pool])
        cbam_feature = self.conv2D(concat)
        return self.multiply([x, cbam_feature])


class DoubleAttention(tensorflow.keras.layers.Layer):
    def __init__(self, name='DoubleAttention', input_channel=32, **kwargs):
        super(DoubleAttention, self).__init__(name=name, **kwargs)

        self.multiply = Multiply()
        self.global_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(input_channel//8, activation='relu',)
        self.dense2 = Dense(input_channel, activation='sigmoid',)

        self.conv2D = Conv2D(filters=1,
                          kernel_size=1,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)
        self.multiply = Multiply()
        self.add = Add()

        # self.concat = Concatenate()


    def call(self, x):
        channel_attention = self.global_pool(x)
        channel_attention = self.dense1(channel_attention)
        channel_attention = self.dense2(channel_attention)
        channel_attention = self.multiply([x, channel_attention])

        spatial_attention = self.conv2D(x)
        spatial_attention = self.multiply([x, spatial_attention])
        return self.add([channel_attention, spatial_attention])


class SpatialAttentionBlock(tensorflow.keras.layers.Layer):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = tensorflow.keras.Sequential([
            Conv2D(in_channels//8,kernel_size=(1,3), padding="same", kernel_initializer='he_normal',),
            BatchNormalization(),
            ReLU()]
        )
        self.key = tensorflow.keras.Sequential([
            Conv2D(in_channels//8, kernel_size=(3,1), padding="same", kernel_initializer='he_normal',),
            BatchNormalization(),
            ReLU()]
        )
        self.value = Conv2D(in_channels, kernel_size=1, kernel_initializer='he_normal')
        # self.gamma = tf.Variable(tf.zeros(1))
        self.softmax = tensorflow.keras.layers.Softmax(axis=-1)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1, ),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, H, W, C = x.shape
        # compress x: [B,H,W, C]-->[B,H*W,C], make a matrix transpose
        proj_query = tf.reshape(self.query(x), (-1, W * H, C))
        proj_key = tf.transpose(tf.reshape(self.key(x), (-1, W * H, C)), (0, 2, 1))
        affinity = tf.linalg.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = tf.transpose(tf.reshape(self.value(x), (-1, H * W, C)), (0, 2, 1))
        weights = tf.linalg.matmul(proj_value, affinity)
        weights = tf.transpose(tf.reshape(weights, (-1, C, H, W,)), (0, 2, 3, 1))
        out = self.gamma * weights + x
        return out

class ChannelAttentionBlock(tensorflow.keras.layers.Layer):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        # self.gamma = tf.Variable(tf.zeros(1))
        self.softmax = tensorflow.keras.layers.Softmax(axis=-1)

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(1, ),
            initializer="random_normal",
            trainable=True,
        )

    def call(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, H, W, C = x.shape
        x = tf.transpose(x, (0, 3, 1, 2))
        proj_query = tf.reshape(x, (-1, C, H * W))
        proj_key = tf.transpose(tf.reshape(x, (-1, C, H * W)), (0, 2, 1))
        affinity = tf.linalg.matmul(proj_query, proj_key)


        affinity_new = tf.math.reduce_max(affinity, -1, keepdims=True) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = tf.reshape(x, (-1, C, H * W))
        weights = tf.linalg.matmul(affinity_new, proj_value)
        weights = tf.reshape(weights, (-1, C, H, W,))

        out = self.gamma * weights + x
        out = tf.transpose(out, (0, 2, 3, 1))

        return out



class AffinityAttention(tensorflow.keras.layers.Layer):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)
        # self.conv1x1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def call(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        out = sab + cab
        return out


class DoubleAttentionSeq(tensorflow.keras.layers.Layer):
    def __init__(self, name='DoubleAttention', input_channel=32, reduction=16, res_connect=False, **kwargs):
        super(DoubleAttentionSeq, self).__init__(name=name, **kwargs)

        self.multiply = Multiply()
        self.global_pool = GlobalAveragePooling2D()
        self.global_max_pool = GlobalMaxPooling2D()

        self.dense1 = Dense(input_channel//reduction, activation='relu',)
        self.dense2 = Dense(input_channel, activation='sigmoid',)
        self.sigmoid = Activation('sigmoid')
        self.res_connect = res_connect
        self.conv2D = Conv2D(filters=1,
                          kernel_size=1,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)
        self.multiply = Multiply()
        self.add = Add()

        self.concat = Concatenate(axis=-1)


    def call(self, x):
        channel_attention = self.global_pool(x)
        channel_attention = self.dense1(channel_attention)
        channel_attention = self.dense2(channel_attention)

        channel_attention_max = self.global_max_pool(x)
        channel_attention_max = self.dense1(channel_attention_max)
        channel_attention_max = self.dense2(channel_attention_max)

        channel_attention = self.add([channel_attention_max, channel_attention])
        channel_attention = self.sigmoid(channel_attention)

        x_channel = self.multiply([x, channel_attention])

        spatial_attention = tf.math.reduce_mean(x_channel, axis=-1, keepdims=True)
        spatial_attention_max = tf.math.reduce_max(x_channel, axis=-1, keepdims=True)

        # spatial_attention = self.global_pool(x_channel)
        # spatial_attention_max = self.global_max_pool(x_channel)

        spatial_attention = self.concat([spatial_attention, spatial_attention_max])
        spatial_attention = self.conv2D(spatial_attention)

        x_channel = self.multiply([x_channel, spatial_attention])

        if self.res_connect:
            x = self.add([x_channel, x])
            return x

        return x_channel

if __name__ == '__main__':
    # tf.compat.v1.enable_eager_execution()
    input_tensor = tf.keras.Input(shape=(64, 64, 32))

    # input_tensor = tf.random.normal(shape=(2, 64, 64, 32))
    example_instance = AffinityAttention(32)
    output_tensor = example_instance(input_tensor)  # calls build
    print(output_tensor.shape)