
import nibabel as nib
import glob
import skimage.transform as skTrans
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import numpy as np
import sys

def generate_3dimage(base_dir="data_folder/train", dim = (320,320,16),
                     batch_size = 2):
    def loadNii(filename):
        img = nib.load(filename)
        data = img.get_fdata()
        return data

    image_base = os.path.join(base_dir, 'images')
    mask_base = os.path.join(base_dir, 'labels')
    files_names = os.listdir(image_base)
    files_names = [i for i in files_names if not i.startswith('.')]
    while True:
        # Select files (paths/indices) for the batch
        batch_names  = np.random.choice(a = files_names, size = batch_size)
        n_channels = 2
        batch_x = np.empty((batch_size, *dim, n_channels))
        batch_y = np.empty((batch_size, *dim))
        # Read in each input, perform preprocessing and get labels
        for i, input_name in enumerate(batch_names):
            image_path = os.path.join(image_base,input_name)
            mask_path = os.path.join(mask_base,input_name)

            input = loadNii(image_path)
            output = loadNii(mask_path)
            input = skTrans.resize(input, dim,
                                   order=1, preserve_range=True)
            output = skTrans.resize(output, dim,
                                   order=1, preserve_range=True)


            batch_x[i] = input
            batch_y[i] = output
        batch_y = np.expand_dims(batch_y, axis=-1)
        yield (batch_x, batch_y)


def generator_slice(base_dir="data_folder/train", dim = (320,320,16), batch_size=128
                     ):
    def _decode_and_resize(file_name, label_name):
        tf.io.imread()

    image_base = os.path.join(base_dir, 'images')
    mask_base = os.path.join(base_dir, 'labels')
    files_names = os.listdir(image_base)
    train_filenames = tf.constant([os.path.join(image_base, i) for i in files_names])
    train_labelnames = tf.constant([os.path.join(mask_base, i) for i in files_names])

    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labelnames))
    train_dataset = train_dataset.map(map_func= _decode_and_resize, 
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset


def read_fn(file_references, mode, params=None):
    # We define a `read_fn` and iterate through the `file_references`, which
    # can contain information about the data to be read (e.g. a file path):
    for meta_data in file_references:

        # Here, we parse the `subject_id` to construct a file path to read
        # an image from.
        subject_id = meta_data[0]
        data_path = '../../data/IXI_HH/1mm'
        t1_fn = os.path.join(data_path, '{}/T1_1mm.nii.gz'.format(subject_id))

        # Read the .nii image containing a brain volume with SimpleITK and get
        # the numpy array:
        sitk_t1 = sitk.ReadImage(t1_fn)
        t1 = sitk.GetArrayFromImage(sitk_t1)

        # Normalise the image to zero mean/unit std dev:
        t1 = whitening(t1)

        # Create a 4D Tensor with a dummy dimension for channels
        t1 = t1[..., np.newaxis]

        # If in PREDICT mode, yield the image (because there will be no label
        # present). Additionally, yield the sitk.Image pointer (including all
        # the header information) and some metadata (e.g. the subject id),
        # to facilitate post-processing (e.g. reslicing) and saving.
        # This can be useful when you want to use the same read function as
        # python generator for deployment.
        if mode == tf.estimator.ModeKeys.PREDICT:
            yield {'features': {'x': t1}}

        # Labels: Here, we parse the class *sex* from the file_references
        # \in [1,2] and shift them to \in [0,1] for training:
        sex = np.int32(meta_data[1]) - 1
        y = sex

        # If training should be done on image patches for improved mixing,
        # memory limitations or class balancing, call a patch extractor
        if params['extract_examples']:
            images = extract_random_example_array(
                t1,
                example_size=params['example_size'],
                n_examples=params['n_examples'])

            # Loop the extracted image patches and yield
            for e in range(params['n_examples']):
                yield {'features': {'x': images[e].astype(np.float32)},
                       'labels': {'y': y.astype(np.int32)}}

        # If desired (i.e. for evaluation, etc.), return the full images
        else:
            yield {'features': {'x': images},
                   'labels': {'y': y.astype(np.int32)}}

    return


# Generator function
def f():
    fn = read_fn(file_references=all_filenames,
                 mode=tf.estimator.ModeKeys.TRAIN,
                 params=reader_params)

    ex = next(fn)
    # Yield the next image
    yield ex


# Timed example with generator io
dataset = tf.data.Dataset.from_generator(
    f, reader_example_dtypes, reader_example_shapes)
dataset = dataset.repeat(None)
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(1)

def generator_pipline(base_dir="data_folder/train", dim = (320,320,16),
                     ):
    def gen():
        def loadNii(filename):
            img = nib.load(filename)
            data = img.get_fdata()
            return data

        image_base = os.path.join(base_dir, 'images')
        mask_base = os.path.join(base_dir, 'labels')
        files_names = os.listdir(image_base)
        files_names = [i for i in files_names if not i.startswith('.')]
        while True:
            # Select files (paths/indices) for the batch
            batch_names  = np.random.choice(a = files_names, size = 1)
            n_channels = 2
            # Read in each input, perform preprocessing and get labels
            input_name  = batch_names[0]
            image_path = os.path.join(image_base,input_name)
            mask_path = os.path.join(mask_base,input_name)

            input = loadNii(image_path)
            output = loadNii(mask_path)
            input = skTrans.resize(input, dim,
                                   order=1, preserve_range=True)
            output = skTrans.resize(output, dim,
                                   order=1, preserve_range=True)

            #output = np.expand_dims(output, axis=-1)

            output = tf.one_hot(output, depth=2,
                       on_value=1, off_value=0,
                       axis=-1)  # output: [4 x 3]
            # input = tf.convert_to_tensor(input, dtype=tf.float32)
            # output = tf.convert_to_tensor(output, dtype=tf.float32)
            #
            # input.set_shape([None, *input.shape])
            # output.set_shape([None, *output.shape])
            yield input, output
    return gen



class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, base_dir="data_folder/train", labels=1, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        #self.labels = labels
        self.image_base = os.path.join(base_dir, 'images')
        self.mask_base = os.path.join(base_dir, 'labels')
        files_names = os.listdir(self.image_base)
        files_names = [i for i in files_names if not i.startswith('.')]
        self.files_names = files_names
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.files_names) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        cur_files_names = self.files_names[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(cur_files_names)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.files_names)

    def __data_generation(self, cur_files_names):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty(self.batch_size, *self.dim)

        for i, input_name in enumerate(cur_files_names):
            image_path = os.path.join(self.image_base,cur_files_names)
            mask_path = os.path.join(self.mask_base,cur_files_names)

            input = nib.load(image_path).get_fdata()
            output = nib.load(mask_path).get_fdata()
            input = skTrans.resize(input, *self.dim,
                                   order=1, preserve_range=True)
            output = skTrans.resize(output, *self.dim,
                                   order=1, preserve_range=True)


            X[i] = input
            y[i] = output

        return X, y


def fixup_shape(images, labels):
    images.set_shape([None, None, None,None,  2])
    labels.set_shape([None, None, None,None, 2]) # I have 19 classes
    return images, labels

def gen_final(path):
    pass