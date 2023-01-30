import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Reshape, Dense, \
    Layer, Flatten, Conv2DTranspose, MaxPooling2D, UpSampling2D, Activation, LeakyReLU,Input
from tensorflow.keras.models import Model

tf.keras.backend.set_floatx('float64')


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        z_mean, z_log_var = inputs

        # eps = tf.random.normal(shape=z_mean.shape)
        # return eps * tf.exp(z_log_var * .5) + z_mean

        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(Layer):
    def __init__(self, latent_dim=2, name='encoder', **kwargs):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()
        self.Flatten = Flatten()
        self.leaky_relu = LeakyReLU()
        self.conv32 = Conv2D(32, 3, strides=1, padding="same")
        self.conv64 = Conv2D(64, 3, strides=1, padding="same")
        self.max_pool = MaxPooling2D(pool_size=(2, 2))
        self.dense_hidden = Dense(1024)

    def call(self, inputs):
        x = self.conv32(inputs)
        x = self.leaky_relu(x)
        x = self.max_pool(x)
        x = self.conv64(x)
        x = self.leaky_relu(x)
        x = self.Flatten(x)
        x = self.dense_hidden(x)
        x = self.leaky_relu(x)

        z_mean = self.dense_mean(x)
        z_mean = self.leaky_relu(z_mean)

        z_log_var = self.dense_log_var(x)
        z_log_var = self.leaky_relu(z_log_var)

        z_sample = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z_sample


class Decoder(Layer):
    def __init__(self, name='decoder', **kwargs):
        super().__init__(name=name)
        self.conv64 = Conv2D(64, 3, strides=1, padding="same")
        self.conv32 = Conv2D(32, 3, strides=1, padding="same")

        self.up_sample = UpSampling2D(size=(2, 2), interpolation='bilinear')

        self.reshape = Reshape((14, 14, 32))

        self.dense_to_meet_size = Dense(14 * 14 * 32)

        self.leaky_relu = LeakyReLU()

        self.conv1D_final = Conv2D(1, 3,
                                   activation="sigmoid",
                                   padding="same")
        self.dense_hidden = Dense(1024)

    def call(self, inputs):
        inputs = self.dense_hidden(inputs)
        x = self.dense_to_meet_size(inputs)
        x = self.leaky_relu(x)
        x = self.reshape(x)
        x = self.conv64(x)
        x = self.leaky_relu(x)
        x = self.up_sample(x)
        x = self.conv32(x)
        x = self.leaky_relu(x)

        z_final = self.conv1D_final(x)

        return z_final


class VariationalAutoEncoder(Model):

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)

        log2pi = tf.cast(log2pi, tf.float64)
        sample = tf.cast(sample, tf.float64)
        mean = tf.cast(mean, tf.float64)
        logvar = tf.cast(logvar, tf.float64)

        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def __init__(self, original_dim, latent_dim=32, name='auto_encoder',
                 w_grad=False,
                 kl_weight=1,
                 **kwargs):
        super().__init__(name=name, )
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim=latent_dim, **kwargs)
        self.decoder = Decoder(**kwargs)
        self.sampling = Sampling()
        self.w_grad = w_grad
        self.kl_weight = kl_weight
        self.kwargs = kwargs
        self.input_dim = None

    # @tf.function
    def call(self, inputs):

        self.input_dim = inputs.shape

        z_mean, z_log_var, z_sample = self.encoder(inputs)
        reconstructed = self.decoder(z_sample)

        # logpx_z = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal_pdf(z_sample, 0., 0.)
        logqz_x = self.log_normal_pdf(z_sample, z_mean, z_log_var)

        kl_loss = tf.reduce_mean(logqz_x - logpz)

        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

        diffs = tf.math.squared_difference(inputs, reconstructed)
        mse_loss = tf.reduce_mean(tf.reduce_sum(diffs, axis=[1, 2, 3]))

        self.add_loss(mse_loss)
        self.add_metric(mse_loss, name="mse_loss")

        kl_loss *= self.kl_weight
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name="kl_loss")

        return reconstructed

    def build_graph(self):
        x = Input(self.input_shape)
        return Model(inputs=[x], outputs=self.call(x))


class VAE_w_grad_only_loss(VariationalAutoEncoder):

    def __init__(self, original_dim, latent_dim=2, name='auto_encoder',
                 kl_weight=0.5, **kwargs):
        super().__init__(name=name, original_dim=original_dim, latent_dim=latent_dim,
                         kl_weight=kl_weight, **kwargs)

    # @tf.function
    def call(self, inputs):  ##TODO: get a tf decrator?

        z_mean, z_log_var, z_sample = self.encoder(inputs)
        reconstructed = self.decoder(z_sample)

        logpz = self.log_normal_pdf(z_sample, 0., 0.)
        logqz_x = self.log_normal_pdf(z_sample, z_mean, z_log_var)

        kl_loss = tf.reduce_mean(logqz_x - logpz)

        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

        diffs = tf.math.squared_difference(inputs, reconstructed)
        mse_loss = tf.reduce_mean(tf.reduce_sum(diffs, axis=[1, 2, 3]))

        self.add_loss(mse_loss)
        self.add_metric(mse_loss, name="mse_loss")

        kl_loss *= self.kl_weight
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name="kl_loss")

        dx, dy = tf.image.image_gradients(inputs)
        dx_recons, dy_recons = tf.image.image_gradients(reconstructed)

        diffs_x = tf.math.squared_difference(dx_recons, dx)
        mse_grad_x = tf.reduce_mean(tf.reduce_sum(diffs_x, axis=[1, 2, 3]))

        diffs_y = tf.math.squared_difference(dy_recons, dy)
        mse_grad_y = tf.reduce_mean(tf.reduce_sum(diffs_y, axis=[1, 2, 3]))

        mse_grad = mse_grad_x + mse_grad_y

        self.add_loss(mse_grad)
        self.add_metric(mse_grad, name="mse_grad")

        return reconstructed


class AutoEncoder(VariationalAutoEncoder):

    def __init__(self, original_dim, latent_dim=32, name='auto_encoder',
                 kl_weight=0.5, **kwargs):
        super().__init__(name=name, original_dim=original_dim, latent_dim=latent_dim,
                         kl_weight=kl_weight, **kwargs)

    # @tf.function
    def call(self, inputs):  ##TODO: get a tf decrator?

        z_mean, z_log_var, z_sample = self.encoder(inputs)
        reconstructed = self.decoder(z_mean)

        diffs = tf.math.squared_difference(inputs, reconstructed)
        mse_loss = tf.reduce_mean(tf.reduce_sum(diffs, axis=[1, 2, 3]))

        self.add_loss(mse_loss)

        self.add_metric(mse_loss, name="mse_loss")

        return reconstructed


class Decoder_for_grad_only(Decoder):
    def __init__(self, name='decoder', **kwargs):
        super().__init__(name=name, **kwargs)
        self.conv1D_final = Conv2D(2, 3,
                                   # activation="sigmoid",
                                   padding="same")

    def call(self, inputs):
        inputs = self.dense_hidden(inputs)
        x = self.dense_to_meet_size(inputs)
        x = self.leaky_relu(x)
        x = self.reshape(x)
        x = self.conv64(x)
        x = self.leaky_relu(x)
        x = self.up_sample(x)
        x = self.conv32(x)
        x = self.leaky_relu(x)

        z_final = self.conv1D_final(x)

        return z_final


class VAE_only_grad(VariationalAutoEncoder):

    def __init__(self, original_dim, latent_dim=2, name='auto_encoder',
                 kl_weight=0.5, **kwargs):
        super().__init__(name=name, original_dim=original_dim, latent_dim=latent_dim,
                         kl_weight=kl_weight, **kwargs)
        self.decoder = Decoder_for_grad_only(**kwargs)

    def call(self, inputs):  ##TODO: get a tf decrator?

        z_mean, z_log_var, z_sample = self.encoder(inputs)
        reconstructed = self.decoder(z_sample)

        logpz = self.log_normal_pdf(z_sample, 0., 0.)
        logqz_x = self.log_normal_pdf(z_sample, z_mean, z_log_var)

        kl_loss = tf.reduce_mean(logqz_x - logpz)

        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)

        kl_loss *= self.kl_weight

        diffs = tf.math.squared_difference(inputs, reconstructed)
        mse_loss = tf.reduce_mean(tf.reduce_sum(diffs, axis=[1, 2, 3]))

        self.add_loss(mse_loss)
        self.add_metric(mse_loss, name="mse_loss")

        kl_loss *= self.kl_weight

        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name=f"kl_loss")

        return reconstructed
