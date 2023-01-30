

from mpunet.logging import ScreenLogger
from mpunet.utils.conv_arithmetics import compute_receptive_fields

from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Input, BatchNormalization, Cropping3D,
                                     Concatenate, Conv3D, MaxPooling3D,
                                     UpSampling3D, Reshape)
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import numpy as np
import sys

from metrics import *
from gens import *

def Unet3dFunctional(img_shape = (320,320,16,2), filters = 64, depth = 3, n_class = 3):
    inputs = Input(shape=img_shape)
    inp = inputs
    residual_connections = []
    for i in range(depth):
        conv = Conv3D(filters=filters,  kernel_size= 3,
                      activation='relu', padding='same')(inp)
        conv = Conv3D(filters=filters,  kernel_size= 3,
                      activation='relu', padding='same')(conv)
        bn = BatchNormalization()(conv)
        inp = MaxPooling3D(pool_size=(2,2,2))(bn)
        filters *=2
        residual_connections.append(bn)

    conv = Conv3D(filters=filters,  kernel_size= 3,
                  activation='relu', padding='same')(inp)
    conv = Conv3D(filters=filters,  kernel_size= 3,
                  activation='relu', padding='same')(conv)
    bn = BatchNormalization()(conv)
    for i in range(depth):
        bn = UpSampling3D(size = (2,2,2))(bn)
        merge = Concatenate(axis=-1)([bn, residual_connections[len(residual_connections)-i-1]])
        filters /=2

        conv = Conv3D(filters=filters,  kernel_size= 3,
                      activation='relu', padding='same')(merge)

        conv = Conv3D(filters=filters,  kernel_size= 3,
                      activation='relu', padding='same')(conv)

        bn = BatchNormalization()(conv)


    out = Conv3D(n_class, 1, activation= 'softmax')(bn)

    model = Model(inputs, out)
    return model

class UNet3D(Model):
    """
    3D UNet implementation with batch normalization and complexity factor adj.

    See original paper at http://arxiv.org/abs/1505.04597
    """
    def __init__(self,
                 n_classes,
                 dim=None,
                 n_channels=1,
                 depth=3,
                 out_activation="softmax",
                 activation="relu",
                 kernel_size=3,
                 padding="same",
                 complexity_factor=1,
                 flatten_output=False,
                 l2_reg=None,
                 logger=None, **kwargs):

        dim1, dim2, dim3 = dim
        # Set various attributes
        self.img_shape = (dim1, dim2, dim3, n_channels)
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
        self.label_crop = np.array([[0, 0], [0, 0], [0, 0]])

        # Build model and init base keras Model class
        super().__init__(*self.init_model())

        # Compute receptive field
        names = [x.__class__.__name__ for x in self.layers]
        index = names.index("UpSampling3D")
        self.receptive_field = compute_receptive_fields(self.layers[:index])[-1][-1]

        # Log the model definition


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
        filters = 2
        in_ = inputs
        residual_connections = []
        for i in range(self.depth):
            conv = Conv3D(int(filters*self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kr)(in_)
            conv = Conv3D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kr)(conv)
            bn = BatchNormalization()(conv)
            in_ = MaxPooling3D(pool_size=(2, 2, 2))(bn)

            # Update filter count and add bn layer to list for residual conn.
            filters *= 2
            residual_connections.append(bn)

        """
        Bottom (no max-pool)
        """
        conv = Conv3D(int(filters * self.cf), self.kernel_size,
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kr)(in_)
        conv = Conv3D(int(filters * self.cf), self.kernel_size,
                      activation=self.activation, padding=self.padding,
                      kernel_regularizer=kr)(conv)
        bn = BatchNormalization()(conv)

        """
        Up-sampling
        """
        residual_connections = residual_connections[::-1]
        for i in range(self.depth):
            # Reduce filter count
            filters /= 2

            # Up-sampling block
            # Note: 2x2 filters used for backward comp, but you probably
            # want to use 3x3 here instead.
            up = UpSampling3D(size=(2, 2, 2))(bn)
            conv = Conv3D(int(filters * self.cf), 2, activation=self.activation,
                          padding=self.padding, kernel_regularizer=kr)(up)
            bn = BatchNormalization()(conv)

            # Crop and concatenate
            cropped_res = self.crop_nodes_to_match(residual_connections[i], bn)
            merge = Concatenate(axis=-1)([cropped_res, bn])

            conv = Conv3D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kr)(merge)
            conv = Conv3D(int(filters * self.cf), self.kernel_size,
                          activation=self.activation, padding=self.padding,
                          kernel_regularizer=kr)(conv)
            bn = BatchNormalization()(conv)

        """
        Output modeling layer
        """
        out = Conv3D(self.n_classes, 1, activation=self.out_activation)(bn)
        if self.flatten_output:
            out = Reshape([np.prod(self.img_shape[:3]),
                           self.n_classes], name='flatten_output')(out)

        return [inputs], [out]

    def crop_nodes_to_match(self, node1, node2):
        """
        If necessary, applies Cropping3D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-1]
        s2 = np.array(node2.get_shape().as_list())[1:-1]

        if np.any(s1 != s2):
            c = (s1 - s2).astype(np.int)
            cr = np.array([c//2, c//2]).T
            cr[:, 1] += c % 2
            cropped_node1 = Cropping3D(cr)(node1)
            self.label_crop += cr
        else:
            cropped_node1 = node1

        return cropped_node1


def my_input_fn(gen, epochs, buffer_size=None):
    dataset = tf.data.Dataset.from_generator(
        generator=gen,
        output_types=(np.float32, np.float32),
    )
    if buffer_size:
        dataset = dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
        dataset = dataset.prefetch(buffer_size=BUFFER_SIZE)
    else:
        dataset = dataset.repeat(epochs)
        dataset = dataset.batch(2)

    return dataset

if __name__ == '__main__':

    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    #dataset = dataset.batch(batch_size=2, drop_remainder=False)

    BUFFER_SIZE = 50
    BATCH_SIZE = 2
    #AUTOTUNE = tf.data.AUTOTUNE
    total_items = 30
    batch_size = 3
    epochs = 10
    num_batches = int(total_items/batch_size)

    sb = generate_3dimage(base_dir=os.path.join(BASE_DIR, "data_folder/train"), batch_size=1)
    #
    gen = generator_pipline(base_dir=os.path.join(BASE_DIR, "data_folder/train"),
                            dim=(320,320,16))

    dataset = my_input_fn(gen, epochs)

    dataset = dataset.map(fixup_shape)

    model = UNet3D(n_classes=2, dim=(320, 320, 16), n_channels=2, depth=2)

    #model = Unet3dFunctional(img_shape=(320, 320, 16, 2), filters=2, depth=2, n_class=2)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  #loss=dice_coef_loss,
                  metrics=['accuracy',
                           iou_back,
                           #BinaryTruePositives(),
                           #tf.keras.metrics.MeanIoU(num_classes=2),
                           CategoricalTruePositives(num_classes=2, batch_size=2),
                           ])

    model.fit_generator(dataset, epochs=epochs,  steps_per_epoch=num_batches)









    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = np.random.rand(1000, 28)
    x_test = np.random.rand(1000, 28)
    y_train = np.random.randint(0,2, size=(1000,1))
    y_test = np.random.randint(0,2, size=(1000,1))

    #@x_train, x_test = x_train / 255.0, x_test / 255.0
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer='adam',
                  loss=loss_fn,
                  metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=1000)

    print(tf.test.is_gpu_available())
    print(sys.path)
    # data = DataGenerator()
    # print(data.__getitem__(0))

    model = Unet3dFunctional(img_shape=(64, 64, 16, 2), filters=2, depth=2, n_class=2)
    model = UNet3D(n_classes=2, dim=16, n_channels=2, depth=2)
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(sb, epochs=5)


# from numba import cuda
# device = cuda.get_current_device()
# device.reset()