"""
Mathias Perslev & Peidi Xu

University of Copenhagen
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.metrics import sparse_categorical_accuracy
from tensorflow.keras.metrics import binary_accuracy

from skimage.metrics import hausdorff_distance

class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', num_classes=3, ignore_zero=True, **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.num_classes = num_classes
        self.ignore_zero = ignore_zero

    def update_state(self, y_true, y_pred, sample_weight=None, smooth=0.01):

        y_true = y_true[0]

        y_pred = tf.math.argmax(y_pred, axis=-1)

        begin = 0
        if self.ignore_zero:
            begin = 1

        dice_all = 0
        for i in range(begin, self.num_classes):

            intersects = tf.math.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i))
            intersects = tf.cast(intersects, tf.uint8)
            intersects = 2 * tf.math.reduce_sum(intersects)
            intersects = tf.cast(intersects, tf.float32)

            s1 = tf.cast(s1, tf.uint8)
            s2 = tf.cast(s2, tf.uint8)

            union = tf.math.reduce_sum(s1) + tf.math.reduce_sum(s2)
            union = tf.cast(union, tf.float32)

            dice = (smooth + 2 * intersects) / (smooth + union)

            dice_all += dice

        dice_all /= (self.num_classes-begin)
        self.true_positives.assign_add(dice_all)

    def result(self):
        return self.true_positives

class MeanIoUManual2Path(tf.keras.metrics.MeanIoU):

    def __init__(self, name='mean_IoU', num_classes=3,  **kwargs):
        self.num_classes = num_classes
        super().__init__(name=name, num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # breakpoint()

        # if y_pred.get
        if y_pred.get_shape()[-1] != self.num_classes:
            y_true = y_true[1]
            y_true = tf.squeeze(y_true)
            y_pred = y_pred > 0.5

        else:
            y_true = y_true[0]
            y_true = tf.squeeze(y_true)
            y_pred = tf.math.argmax(y_pred, axis=-1)

        # if len(y_pred.get_shape()) != len(y_true.get_shape()):
        #     return 0

        super().update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)



class MeanIoUManual(tf.keras.metrics.MeanIoU):

    def __init__(self, name='mean_IoU', num_classes=3,  **kwargs):
        self.num_classes = num_classes
        super().__init__(name=name, num_classes=num_classes)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = tf.squeeze(y_true)
        y_pred = tf.math.argmax(y_pred, axis=-1)

        super().update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class MeanIoUManualBinaryTwoPath(tf.keras.metrics.MeanIoU):

    def __init__(self, name='mean_IoU_Binary', num_classes=1,  **kwargs):
        super().__init__(name=name, num_classes=1)

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_true = y_true[1]
        y_true = tf.squeeze(y_true)
        y_pred = y_pred > 0.5

        super().update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)


class MeanIoUManualBinary(tf.keras.metrics.MeanIoU):

    def __init__(self, name='mean_IoU_Binary', num_classes=1,  **kwargs):
        super().__init__(name=name, num_classes=2)

    def update_state(self, y_true, y_pred, sample_weight=None):

        # y_true = tf.squeeze(y_true)
        y_pred = y_pred > 0.5

        super().update_state(y_true=y_true, y_pred=y_pred, sample_weight=sample_weight)



def two_path_sparse_categorical_accuracy_manual(y_true, y_pred):
    return sparse_categorical_accuracy(y_true[0], y_pred)

def two_path_sparse_categorical_dice_manual(y_true, y_pred):
    return dice_all_tf(y_true[0], y_pred)



def two_path_binary_accuracy_manual(y_true, y_pred):
    return binary_accuracy(y_true[1], tf.expand_dims(y_pred, axis=-1))


def empty(y_true, y_pred):
    return 0

def assd_surface(y_true, y_pred):

    from scipy.ndimage import distance_transform_edt

    # Flatten and bool the ground truth and predicted labels
    s1 = np.array(y_true > 0).astype('bool')
    s2 = np.array(y_pred > 0).astype('bool')

    # label_ids = sorted(np.unique(s1))[1:]
    # distances = np.zeros((s1.shape[0], s1.shape[1], s1.shape[2], len(label_ids)))
    #
    # distances = np.zeros((s1.shape[0], s1.shape[1], s1.shape[2]))
    #
    # reserve = s1 * s2

    distances_1 = distance_transform_edt(1-s2) * (s1 != 0)
    distances_2 = distance_transform_edt(1-s1) * (s2 != 0)
    
    mean_distance_1 = np.sum(distances_1)/np.sum(s1 != 0)
    mean_distance_2 = np.sum(distances_2)/np.sum(s2 != 0)

    # print(f'mean assd true from pred = {mean_distance_1}')
    #
    # print(f'mean assd pred from true = {mean_distance_2}')
    #
    # print(f'mean assd  = {(mean_distance_1 + mean_distance_2)/2}')

    return (mean_distance_1 + mean_distance_2)/2

    return distances_1>0, distances_2>0


def cal_gap(y_true, y_pred, dist=10, use_predict=True):
    from scipy.ndimage import distance_transform_edt
    from mpunet.preprocessing.weight_map import unet_weight_map3D, get_distance

    # Flatten and bool the ground truth and predicted labels

    weight_map_s1 = get_distance(y_true) < dist

    if use_predict:
        weight_map_s2 = get_distance(y_pred) < dist
        mask = np.logical_or(weight_map_s1, weight_map_s2)
    else:
        return weight_map_s1

    return mask



def dice(y_true, y_pred, smooth=1.0, mask=None):
    """
    Calculates the Soerensen dice coefficient between two binary sets
    """
    # Flatten and bool the ground truth and predicted labels
    s1 = np.array(y_true).flatten().astype(np.bool)
    s2 = np.array(y_pred).flatten().astype(np.bool)

    if mask is not None:
        mask = mask.flatten().astype(np.bool)
        s1 = np.logical_and(s1, mask)
        s2 = np.logical_and(s2, mask)
    #
    # (s1 & s2 & mask).sum()
    # (s1 & mask).sum()
    # Calculate dice
    return (smooth + 2 * np.logical_and(s1, s2).sum()) / (smooth +
                                                          s1.sum() + s2.sum())


def dice_tf(y_true, y_pred, smooth=1.0):
    """
    Calculates the Soerensen dice coefficient between two binary sets
    """
    # Flatten and bool the ground truth and predicted labels
    s1 = tf.reshape(y_true, [-1])
    s2 = tf.reshape(y_pred, [-1])
    s1 = tf.cast(s1, tf.bool)
    s2 = tf.cast(s2, tf.bool)

    # Calculate dice
    return (smooth + 2 * tf.math.reduce_sum(tf.math.logical_and(s1, s2))) \
           / (smooth + tf.math.reduce_sum(s1) + tf.math.reduce_sum(s2))


def dice_all_tf(y_true, y_pred, smooth=1.0, n_classes=None, ignore_zero=True,
             skip_if_no_y=False):
    """
    Calculates the Soerensen dice coefficients for all unique classes
    """
    # Get array of unique classes in true label array
    # breakpoint()
    if n_classes is None:
        shape = y_pred.get_shape()
        n_classes = shape[-1]
    # Ignore background class?

    begin = 0
    if ignore_zero:
        begin = 1

    # Calculate dice for all targets
    dice_coeffs = 0
    for _class in range(begin, n_classes):
        s1 = y_true == _class
        if skip_if_no_y and not np.any(s1):
            continue
        s2 = y_pred == _class

        if tf.math.reduce_any(s1) or tf.math.reduce_any(s2):
            d = dice_tf(s1, s2, smooth=smooth)
            dice_coeffs += d
    return dice_coeffs/(n_classes-begin)

def _get_shapes_and_one_hot(y_true, y_pred):
    shape = y_pred.get_shape()
    n_classes = shape[-1]
    # Squeeze dim -1 if it is == 1, otherwise leave it
    dims = tf.cond(tf.equal(y_true.shape[-1] or -1, 1), lambda: tf.shape(y_true)[:-1], lambda: tf.shape(y_true))
    y_true = tf.reshape(y_true, dims)
    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=n_classes)
    return y_true, shape, n_classes

@tf.function
def sparse_dice_score(y_true, y_pred, smooth=0.01, n_classes=None, ignore_zero=True):

    y_pred = tf.math.argmax(y_pred, axis=-1)
    y_true = y_true[0]

    if n_classes is None:
        shape = y_pred.get_shape()
        n_classes = shape[-1]
    # Ignore background class?

    begin = tf.constant(0)
    if ignore_zero:
        begin = tf.constant(1)

    n_classes = tf.constant(n_classes)

    dice_all = tf.constant(0)

    i = begin
    while tf.less(i, n_classes):
        intersects = tf.math.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i))
        intersects = tf.cast(intersects, tf.uint8)
        intersects = 2 * tf.math.reduce_sum(intersects)
        intersects = tf.cast(intersects, tf.float32)

        s1 = tf.equal(y_true, i)
        s1 = tf.cast(s1, tf.uint8)

        s2 = tf.equal(y_pred, i)
        s2 = tf.cast(s2, tf.uint8)

        union = tf.math.reduce_sum(s1) + tf.math.reduce_sum(s2)
        union = tf.cast(union, tf.float32)

        dice = (smooth + 2 * intersects) / (smooth + union)

        i = tf.add(i, 1)

        dice_all = tf.add(dice_all, dice)

    return dice_all/(n_classes-begin)

def dice_single(y_true, y_pred, i, smooth=0.01):
    intersects = tf.math.logical_and(tf.equal(y_true, i), tf.equal(y_pred, i))
    intersects = tf.cast(intersects, tf.uint8)
    intersects = 2 * tf.math.reduce_sum(intersects)
    intersects = tf.cast(intersects, tf.float32)

    s1 = tf.equal(y_true, i)
    s1 = tf.cast(s1, tf.uint8)

    s2 = tf.equal(y_pred, i)
    s2 = tf.cast(s2, tf.uint8)

    union = tf.math.reduce_sum(s1) + tf.math.reduce_sum(s2)
    union = tf.cast(union, tf.float32)

    dice = (smooth + 2 * intersects) / (smooth + union)

    return dice


def dice_all(y_true, y_pred, smooth=1.0, n_classes=None, ignore_zero=True,
             skip_if_no_y=False, mask=None):
    """
    Calculates the Soerensen dice coefficients for all unique classes
    """
    # Get array of unique classes in true label array
    if n_classes is None:
        classes = np.unique(y_true)
    else:
        classes = np.arange(max(2, n_classes))
    # Ignore background class?
    if ignore_zero:
        classes = classes[np.where(classes != 0)]

    # Calculate dice for all targets
    dice_coeffs = np.empty(shape=classes.shape, dtype=np.float32)
    dice_coeffs.fill(np.nan)
    for idx, _class in enumerate(classes):
        s1 = y_true == _class
        if skip_if_no_y and not np.any(s1):
            continue
        s2 = y_pred == _class

        if np.any(s1) or np.any(s2):
            d = dice(s1, s2, smooth=smooth, mask=mask)
            dice_coeffs[idx] = d
    return dice_coeffs


def hausdorff_distance_all(y_true, y_pred, n_classes=None, ignore_zero=True,
             skip_if_no_y=False):
    """
    Calculates the Soerensen dice coefficients for all unique classes
    """
    # Get array of unique classes in true label array
    if n_classes is None:
        classes = np.unique(y_true)
    else:
        classes = np.arange(max(2, n_classes))
    # Ignore background class?
    if ignore_zero:
        classes = classes[np.where(classes != 0)]

    # Calculate dice for all targets
    dice_coeffs = np.empty(shape=classes.shape, dtype=np.float32)
    dice_coeffs.fill(np.nan)
    for idx, _class in enumerate(classes):
        s1 = y_true == _class
        if skip_if_no_y and not np.any(s1):
            continue
        s2 = y_pred == _class

        if np.any(s1) or np.any(s2):
            d =  hausdorff_distance(s1, s2)
            dice_coeffs[idx] = d

    return dice_coeffs

def class_wise_kappa(true, pred, n_classes=None, ignore_zero=True):
    from sklearn.metrics import cohen_kappa_score
    if n_classes is None:
        classes = np.unique(true)
    else:
        classes = np.arange(max(2, n_classes))
    # Ignore background class?
    if ignore_zero:
        classes = classes[np.where(classes != 0)]

    # Calculate kappa for all targets
    kappa_scores = np.empty(shape=classes.shape, dtype=np.float32)
    kappa_scores.fill(np.nan)
    for idx, _class in enumerate(classes):
        s1 = true == _class
        s2 = pred == _class

        if np.any(s1) or np.any(s2):
            kappa_scores[idx] = cohen_kappa_score(s1, s2)
    return kappa_scores


def one_class_dice(y_true, y_pred, smooth=1.0):
    # Predict
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    return (smooth + 2.0 * tf.reduce_sum(y_true * y_pred)) / (smooth + tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))


def sparse_fg_recall(y_true, y_pred, bg_class=0):
    # Get MAP estimates
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), tf.int32)

    # Remove background
    mask = tf.not_equal(y_true, bg_class)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def sparse_mean_fg_f1(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)

    # Get confusion matrix
    cm = tf.confusion_matrix(tf.reshape(y_true, [-1]),
                             tf.reshape(y_pred, [-1]))

    # Get precisions
    TP = tf.diag_part(cm)
    precisions = TP / tf.reduce_sum(cm, axis=0)

    # Get recalls
    TP = tf.diag_part(cm)
    recalls = TP / tf.reduce_sum(cm, axis=1)

    # Get F1s
    f1s = (2 * precisions * recalls) / (precisions + recalls)

    return tf.reduce_mean(f1s[1:])


def sparse_mean_fg_precision(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)

    # Get confusion matrix
    cm = tf.confusion_matrix(tf.reshape(y_true, [-1]),
                             tf.reshape(y_pred, [-1]))

    # Get precisions
    TP = tf.diag_part(cm)
    precisions = TP / tf.reduce_sum(cm, axis=0)

    return tf.reduce_mean(precisions[1:])


def sparse_mean_fg_recall(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)

    # Get confusion matrix
    cm = tf.confusion_matrix(tf.reshape(y_true, [-1]),
                             tf.reshape(y_pred, [-1]))

    # Get precisions
    TP = tf.diag_part(cm)
    recalls = TP / tf.reduce_sum(cm, axis=1)

    return tf.reduce_mean(recalls[1:])


def sparse_fg_precision(y_true, y_pred, bg_class=0):
    # Get MAP estimates
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(tf.reshape(tf.argmax(y_pred, axis=-1), [-1]), tf.int32)

    # Remove background
    mask = tf.not_equal(y_pred, bg_class)
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)

    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))

