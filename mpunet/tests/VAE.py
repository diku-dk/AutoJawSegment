import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, BatchNormalization, Cropping2D, \
    Concatenate, Conv2D, MaxPooling2D, \
    UpSampling2D, Reshape, GlobalAveragePooling2D, Dense, Layer
from tensorflow import keras

#%%
w_init = tf.random_normal_initializer()
w = tf.Variable(initial_value=w_init(shape=(2, 2), dtype="float32")
                             , trainable=True)
# w.assign((np.ones((2,2))))
# w.assign_add((np.ones((2,2))))
x = tf.constant([[3.], [4.]])
x = tf.constant([[3.,4.]],shape=(2,1))
"""1 has to be specific here"""


print(f'w= {w.value()}')
print(f'wx= {tf.matmul(w,x)}')
#%%

class Linear(Layer):
    def __init__(self, units=32, input_dim=32):
        super().__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32")
                             , trainable=True)
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype="float32")
                             , trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class ComputeSum(Layer):
    def __init__(self, input_dim):
        super(ComputeSum, self).__init__()
        self.total = tf.Variable(initial_value=tf.zeros((input_dim,)), trainable=False)

    def call(self, inputs):
        #return tf.reduce_sum(inputs, axis=0)
        self.total.assign_add(tf.reduce_sum(inputs, axis=0))
        return self.total


x = tf.ones((6, 2))
linear_layer = Linear(4, 2)
y = linear_layer(x)
print(y)
my_sum = ComputeSum(2)
y = my_sum(x)
print(y.numpy())
y = my_sum(x)
print(y.numpy())
#%%
class Linear_w_build(Layer):
    def __init__(self, units=32):
        super().__init__()
        self.units = units
    def build(self,input_shape):
        w_init = tf.random_normal_initializer()
        # self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32")
        #                      , trainable=True)
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units), initializer="random_normal", trainable=True
        )
        # b_init = tf.zeros_initializer()
        # self.b = tf.Variable(initial_value=b_init(shape=(self.units,), dtype="float32")
        #                      , trainable=True)
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
#%%
linear_layer = Linear_w_build(10)
x = tf.ones((6,2))
y = tf.ones((6,10))
opt = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.MeanAbsoluteError()

# #%%
# inputs = tf.keras.Input(shape=(2,))
# outputs = linear_layer(inputs)
# model = tf.keras.Model(inputs,outputs)
#
# #%%
# model.compile(optimizer="adam",loss ='mse')
# model.fit(x,y)
# #%%
# for _ in range(10):
#     with tf.GradientTape() as tape:
#         logits = linear_layer(x)
#         loss_value = loss_fn(y,logits)
#
#         grads = tape.gradient(loss_value, model.trainable_weights)
#         opt.apply_gradients(zip(grads, model.trainable_weights))

#%%

class Sampling(Layer):
    # def __int__(self,**kwargs):
    #     super(Sampling, self).__int__(**kwargs)
    def call(self,inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5*z_log_var) * epsilon

class Encoder(Layer):
    def __init__(self,latent_dim=32,hidden_dim=64,name='encoder',**kwargs):
        super(Encoder, self).__init__(name = name,**kwargs)
        self.dense_hidden = Dense(hidden_dim,activation='relu')
        self.dense_mean = Dense(latent_dim)
        self.dense_log_var = Dense(latent_dim)
        self.sampling = Sampling()

    def call(self,inputs):
        x = self.dense_hidden(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z_sample = self.sampling((z_mean,z_log_var))
        return z_mean,z_log_var,z_sample

class Decoder(Layer):
    def __init__(self,final_dim=32,hidden_dim=64,name='encoder',**kwargs):
        super().__init__(name = name, **kwargs)
        self.dense_hidden = Dense(hidden_dim,activation='relu')
        self.dense_final = Dense(final_dim,activation='sigmoid')

    def call(self,inputs):
        x = self.dense_hidden(inputs)
        z_final = self.dense_final(x)
        return z_final
#%%

#%%
class VariationalAutoEncoder(Model):
    def __init__(self,original_dim,hidden_dim=64,latent_dim=32,name='auto_encoder',**kwargs):
        super().__init__(name = name,**kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(latent_dim,hidden_dim)
        self.decoder = Decoder(original_dim,hidden_dim)
        self.sampling = Sampling()

    def call(self,inputs):
        z_mean, z_log_var, z_sample = self.encoder(inputs)
        reconstructed = self.decoder(z_sample)
        kl_loss = -0.5 * tf.reduce_mean(z_log_var-tf.square(z_mean)-tf.exp(z_log_var)+1)
        self.add_loss(kl_loss)
        self.add_metric(kl_loss, name="kl_loss")
        self.add_metric(self.losses, name="total_loss")

        return reconstructed
    # @property
    # def metrics(self):
    #     return [
    #         self.total_loss_tracker,
    #         self.reconstruction_loss_tracker,
    #         self.kl_loss_tracker,
    #     ]

#%%
original_dim = 784
vae = VariationalAutoEncoder(original_dim=original_dim,hidden_dim=64,latent_dim=32)
opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
mse_loss_fn = tf.keras.losses.MeanSquaredError()
loss_metric = tf.keras.metrics.Mean()
(x_train,y_train), (x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255
x_test = x_train.reshape(-1, 784).astype("float32") / 255

train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(64)
epochs = 2
#%%
#vae.summary()

#%%

for ep in range(epochs):
    print(f'starting of epoch {ep}')
    for step, x_batch in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            logits = vae(x_batch)
            mse_loss_value = mse_loss_fn(x_batch,logits)
            loss = mse_loss_value + sum(vae.losses)

            grads = tape.gradient(loss, vae.trainable_weights)
            opt.apply_gradients(zip(grads, vae.trainable_weights))

            loss_metric(loss)

            if step % 100 == 0:
                print("step %d: mean loss = %.4f" % (step, loss_metric.result()))

#%%
plot_label_clusters(vae, x_train, y_train)
#%%
vae = VariationalAutoEncoder(784, 64, 32)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

vae.compile(optimizer,
            loss=tf.keras.losses.BinaryCrossentropy()
            )
vae.fit(x_train, x_train, epochs=10, batch_size=64)
#%%
from tensorflow.keras import layers
original_dim = 784
intermediate_dim = 64
latent_dim = 32

# Define encoder model.
original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
x = layers.Dense(intermediate_dim, activation="relu")(original_inputs)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()((z_mean, z_log_var))
encoder = tf.keras.Model(inputs=original_inputs, outputs=z, name="encoder")

# Define decoder model.
latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
x = layers.Dense(intermediate_dim, activation="relu")(latent_inputs)
outputs = layers.Dense(original_dim, activation="sigmoid")(x)
decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

# Define VAE model.
outputs = decoder(z)
vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")

# Add KL divergence regularization loss.
kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
vae.add_loss(kl_loss)

# Train.
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
vae.fit(x_train, x_train, epochs=10, batch_size=128)
#%%
import matplotlib.pyplot as plt
def vis_result_pair(x_pred_train,n=10):
    plt.figure(figsize=(20, 2))
    for i in range(1, n + 1):
        index = np.random.randint(low=0, high=x_pred_train.shape[0])
        ax = plt.subplot(2, n, i)
        ax.imshow(x_pred_train[index].reshape(28, 28))
        plt.gray()

        ax = plt.subplot(2, n, n+i)
        ax.imshow(x_train[index].reshape(28, 28))
        plt.gray()

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
#%%
x_pred_train = vae.predict(x_train)
vis_result_pair(x_pred_train,n=10)

#%%
def plot_label_clusters(vae, data, labels):
    z_mean, _, _ = vae.encoder(data)
    plt.figure(figsize=(6, 6))
    z_mean = z_mean.numpy()
    # m,n = z_mean.shape
    # z_mean = z_mean.reshape((m,4,8))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.show()
#%%
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255

plot_label_clusters(vae, x_train, y_train)
#%%
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

latent_dim = 2

encoder_inputs = keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(16, activation="relu")(x)
z_mean = layers.Dense(latent_dim, name="z_mean")(x)
z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()
#%%
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(7 * 7 * 64, activation="relu")(latent_inputs)
x = layers.Reshape((7, 7, 64))(x)
x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()
#%%
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
#%%
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
mnist_digits = np.concatenate([x_train, x_test], axis=0)
mnist_digits = np.expand_dims(mnist_digits, -1).astype("float32") / 255

vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())
vae.fit(mnist_digits, epochs=2, batch_size=128)
#%%
import matplotlib.pyplot as plt
# Display a 2D manifold of the digits
n = 15  # figure with 15x15 digits
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# We will sample n points within [-15, 15] standard deviations
grid_x = np.linspace(-15, 15, n)
grid_y = np.linspace(-15, 15, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()


#%%
print('build methods')
x = tf.ones((6, 2))
linear_layer = Linear_w_build(4)
y = linear_layer(x)
print(y)

class Unet(Model):
    def __init__(self, num_classes=1000):
        super(Unet, self).__init__()
        self.block_1 = ResNetBlock()
        self.block_2 = ResNetBlock()
        self.global_pool = GlobalAveragePooling2D()
        self.classifier = Dense(num_classes)

    def call(self, inputs):
        x = self.block_1(inputs)
        x = self.block_2(x)
        x = self.global_pool(x)
        return self.classifier(x)
