import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def dice_coef(y_true, y_pred, smooth=0.1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


class BinaryTruePositives(tf.keras.metrics.Metric):

    def __init__(self, name='binary_true_positives', **kwargs):
        super(BinaryTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # y_true = tf.cast(y_true, tf.bool)
        # y_pred = tf.cast(y_pred, tf.bool)

        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        smooth = 0.0001
        self.true_positives.assign_add((2. * intersection + smooth) / (K.sum(y_true_f) +
                                                                       K.sum(y_pred_f) + smooth))

        # if sample_weight is not None:
        #   sample_weight = tf.cast(sample_weight, self.dtype)
        #   sample_weight = tf.broadcast_to(sample_weight, values.shape)
        #   values = tf.multiply(values, sample_weight)
        # self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives


import tensorflow.keras.backend as K


def intersection_over_union(labels, prediction):
    pred = tf.argmax(prediction, axis=3)
    labl = tf.constant(labels)
    m = tf.metrics.MeanIoU(num_classes=3)
    m.update_state(labl, pred)

    return m.result().numpy()


def dice_back(labels, prediction):
    pred = K.argmax(prediction, axis=3)

    labels = K.cast(labels, 'float32')
    pred = K.cast(pred, 'float32')

    y_true_f = K.flatten(labels)
    y_pred_f = K.flatten(pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 0.00001
    res = (2 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return res


def iou_back(labels, prediction):
    pred = K.argmax(prediction, axis=-1)
    if labels.shape[-1] == 1:  # if not one-hot
        labels = K.squeeze(labels, axis=-1)
    else:
        labels = K.argmax(labels, axis=-1)
    # labels = K.cast(labels,'float32')
    # pred = K.cast(pred,'float32')

    mIoU = 0
    n_class = prediction.shape[-1]
    for i in range(n_class):
        y_pred_f = K.cast(K.equal(pred, i), 'float32')
        y_true_f = K.cast(K.equal(labels, i), 'float32')

        intersection = K.sum(y_true_f * y_pred_f)
        smooth = 0.00001
        union = K.sum(y_true_f + y_pred_f) - intersection

        res = (intersection + smooth) / (union + smooth)
        mIoU += res

    return mIoU / n_class


class ConfusionMatrixMetric(tf.keras.metrics.Metric):
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total_cm.assign_add(self.confusion_matrix(y_true, y_pred))
        return self.total_cm

    def result(self):
        return self.process_confusion_matrix()

    def confusion_matrix(self, y_true, y_pred):
        """
        Make a confusion matrix
        """
        y_pred = tf.argmax(y_pred, 1)
        cm = tf.math.confusion_matrix(y_true, y_pred, dtype=tf.float32, num_classes=self.num_classes)
        return cm

    def process_confusion_matrix(self):
        "returns precision, recall and f1 along with overall accuracy"
        cm = self.total_cm
        diag_part = tf.linalg.diag_part(cm)
        precision = diag_part / (tf.reduce_sum(cm, 0) + tf.constant(1e-15))
        recall = diag_part / (tf.reduce_sum(cm, 1) + tf.constant(1e-15))
        f1 = 2 * precision * recall / (precision + recall + tf.constant(1e-15))
        return precision, recall, f1


from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback


class IntervalEvaluation(Callback):
    def __init__(self, interval=10):
        super(Callback, self).__init__()
        self.interval = interval
        # self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            X_val, self.y_val = self.validation_data

            y_pred = self.model.predict_proba(self.X_val, verbose=0)

            labels = self.y_val
            n_class = y_pred.shape[-1]
            pred = np.argmax(y_pred, axis=-1)

            if labels.shape[-1] == 1:  # if not one-hot
                labels = K.squeeze(labels, axis=-1)
            else:
                labels = K.argmax(labels, axis=-1)

            mIoU = 0
            for i in range(n_class):
                y_pred_f = pred == i
                y_true_f = labels == i

                intersection = np.sum(y_true_f * y_pred_f)
                smooth = 0.00001
                union = np.sum(y_true_f + y_pred_f) - intersection

                res = (intersection + smooth) / (union + smooth)
                mIoU += res

            score = res / n_class
            print("interval evaluation - epoch: {:d} - score: {:.6f}".format(epoch, score))


class CategoricalTruePositives(tf.keras.metrics.Metric):
    def __init__(self, num_classes, batch_size,
                 name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.cat_true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        n_class = y_pred.shape[-1]
        y_pred = K.argmax(y_pred, axis=-1)

        if y_true.shape[-1] == 1:  # if not one-hot
            labels = K.squeeze(y_true, axis=-1)
        else:
            labels = K.argmax(y_true, axis=-1)

        mIoU = 0
        for i in range(n_class):
            y_pred_f = K.cast(K.equal(y_pred, i), 'float32')
            y_true_f = K.cast(K.equal(labels, i), 'float32')

            intersection = K.sum(y_true_f * y_pred_f)
            smooth = 0.00001
            union = K.sum(y_true_f + y_pred_f) - intersection

            res = (intersection + smooth) / (union + smooth)
            mIoU += res

        self.cat_true_positives.assign_add(mIoU / n_class)

    def result(self):
        return self.cat_true_positives
