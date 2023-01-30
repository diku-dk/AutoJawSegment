import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
    Concatenate, Conv2D, MaxPooling2D, \
    UpSampling2D, Reshape, GlobalAveragePooling2D, Dense, Layer, Flatten, Conv2DTranspose
from tensorflow import keras
import os
import sys
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs

        # eps = tf.random.normal(shape=z_mean.shape)
        # return eps * tf.exp(z_log_var * .5) + z_mean

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Encoder(Layer):
    def __init__(self,latent_dim=32,hidden_dim=64,name='encoder', **kwargs):
        super(Encoder, self).__init__(name = name,**kwargs)

        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()
        self.Flatten = Flatten()
        self.conv32 = Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv64 = Conv2D(64, 3, activation="relu", strides=2, padding="same")

    def call(self,inputs):
        x = self.conv32(inputs)
        x = self.conv64(inputs)

        x = self.Flatten(x)

        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z_sample = self.sampling((z_mean,z_log_var))
        return z_mean,z_log_var,z_sample

class Decoder(Layer):
    def __init__(self,final_dim=32,hidden_dim=64,name='decoder',**kwargs):
        super().__init__(name = name, **kwargs)
        self.convTranspose32 = Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.convTranspose64 = Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.reshape = Reshape((7, 7, 32))

        self.dense_to_meet_size = Dense(7 * 7 * 32, activation="relu")

        self.conv1D_final = Conv2DTranspose(1, 3,
                                            #activation="sigmoid",
                                            padding="same")

    def call(self,inputs):
        x = self.dense_to_meet_size(x)
        x = self.reshape(x)
        x = self.convTranspose64(x)
        x = self.convTranspose32(x)
        z_final = self.conv1D_final(x)

        return z_final

class VariationalAutoEncoder(Model):

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def __init__(self,original_dim,hidden_dim=64,latent_dim=32,name='auto_encoder', **kwargs):
        super().__init__(name = name,**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim,hidden_dim=hidden_dim, is_conv=is_conv)
        self.decoder = Decoder(final_dim=original_dim,hidden_dim=hidden_dim, is_conv=is_conv)
        self.sampling = Sampling()

    def call(self,inputs):
        z_mean, z_log_var, z_sample = self.encoder(inputs)
        reconstructed = self.decoder(z_sample)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstructed, labels=inputs)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z_sample, 0., 0.)
        logqz_x = self.log_normal_pdf(z_sample, z_mean, z_log_var)

        self.add_loss(-tf.reduce_mean(logpx_z + logpz - logqz_x))

        #mse_loss = tf.keras.losses.MeanSquaredError()(inputs, reconstructed)
        mse_loss = tf.keras.losses.BinaryCrossentropy()(inputs, reconstructed)

        #self.add_loss(mse_loss)

        #self.add_metric(kl_loss, name="kl_loss")
        #self.add_metric(mse_loss, name="mse_loss")

        self.add_metric(self.losses, name="total_loss")

        return reconstructed