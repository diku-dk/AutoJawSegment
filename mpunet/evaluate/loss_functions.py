import tensorflow as tf
from tensorflow.python.keras.losses import LossFunctionWrapper
from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.distribute import distribution_strategy_context
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops.losses import losses_impl
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import keras_export
from tensorflow.tools.docs import doc_controls
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.losses import binary_crossentropy

def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.

    OBS: Code implemented by Tensorflow

    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.

    # Returns
        A tensor.
    """
    x = tf.convert_to_tensor(x)
    if x.dtype != dtype:
        x = tf.cast(x, dtype)
    return x


def _get_shapes_and_one_hot(y_true, y_pred):
    shape = y_pred.get_shape()
    n_classes = shape[-1]
    # Squeeze dim -1 if it is == 1, otherwise leave it
    dims = tf.cond(tf.equal(y_true.shape[-1] or -1, 1), lambda: tf.shape(y_true)[:-1], lambda: tf.shape(y_true))
    y_true = tf.reshape(y_true, dims)
    y_true = tf.one_hot(tf.cast(y_true, tf.uint8), depth=n_classes)
    return y_true, shape, n_classes


def sparse_jaccard_distance_loss(y_true, y_pred, smooth=1):
    """
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))

    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.

    Approximates the class-wise jaccard distance computed per-batch element
    across spatial image dimensions. Returns the 1 - mean(per_class_distance)
    for each batch element.

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:

    Ref: https://en.wikipedia.org/wiki/Jaccard_index

    @url: https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    @author: wassname
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    sum_ = tf.reduce_sum(y_true + y_pred, axis=reduction_dims)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return 1.0 - tf.reduce_mean(jac, axis=-1, keepdims=True)


class SparseJaccardDistanceLoss(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_jaccard_distance_loss """
    def __init__(self,
                 reduction,
                 smooth=1,
                 name='sparse_jaccard_distance_loss',
                 **kwargs):
        super(SparseJaccardDistanceLoss, self).__init__(
            sparse_jaccard_distance_loss,
            name=name,
            reduction=reduction,
            smooth=smooth
        )


def sparse_dice_loss(y_true, y_pred, smooth=0.01):
    """
    Approximates the class-wise dice coefficient computed per-batch element
    across spatial image dimensions. Returns the 1 - mean(per_class_dice) for
    each batch element.

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    # _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
    # y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)


    intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    union = tf.reduce_sum(y_true + y_pred, axis=reduction_dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice, axis=-1, keepdims=True)


class SparseDiceLoss(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 reduction,
                 smooth=1,
                 name='sparse_dice_loss',
                 **kwargs):
        super(SparseDiceLoss, self).__init__(
            sparse_dice_loss,
            name=name,
            reduction=reduction,
            smooth=smooth
        )

def combined_binary_entropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_true_binary = y_true > 0 # un-one_hot
    y_pred_binary = tf.reduce_sum(y_pred[..., 1:], axis=-1, keepdims=True)

    be = binary_crossentropy(y_true_binary, y_pred_binary, from_logits=from_logits, label_smoothing=label_smoothing)
    ce = sparse_categorical_crossentropy(y_true, y_pred, from_logits)

    return be + ce

def ce_manual_legacy(y_true, y_pred, smooth=1):

    """
    Approximates the class-wise dice coefficient computed per-batch element
    across spatial image dimensions. Returns the 1 - mean(per_class_dice) for
    each batch element.

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    union = tf.reduce_sum(y_true + y_pred, axis=reduction_dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice, axis=-1, keepdims=True)


def ce_manual(y_true, y_pred, from_logits=False, label_smoothing=0):
    return sparse_categorical_crossentropy(y_true, y_pred, from_logits)


class CrossEntropy_Manual(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 reduction,
                 name='ce_manual',
                 from_logits=False,
                 **kwargs):
        super(CrossEntropy_Manual, self).__init__(
            ce_manual,
            name=name,
            reduction=reduction,
            from_logits=from_logits
        )


def two_path_loss(y_true, y_pred, from_logits=False, label_smoothing=0):
    breakpoint()
    print(f'y_true shape = {y_true.shape}')
    print(f'y_pred shape = {y_pred.shape}')
    
    cat_loss = sparse_categorical_crossentropy(y_true[0], y_pred, from_logits)
    sur_loss = binary_crossentropy(y_true, y_pred, from_logits)
    return cat_loss + sur_loss

class TwoPathLoss(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 reduction,
                 name='ce_manual',
                 from_logits=False,
                 **kwargs):
        super().__init__(
            two_path_loss,
            name=name,
            reduction=reduction,
            from_logits=from_logits
        )


class TwoPathSparseCE(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_dice_loss """

    def __init__(self,
                 reduction,
                 name='2ce_manual',
                 from_logits=False,
                 **kwargs):
        super().__init__(
            two_path_sparse_ce,
            name=name,
            reduction=reduction,
            from_logits=from_logits
        )

def two_path_sparse_ce(y_true, y_pred, from_logits=False):
    # breakpoint()
    return sparse_categorical_crossentropy(y_true[0], y_pred, from_logits)

def two_path_bce(y_true, y_pred, from_logits=False, label_smoothing=0):
    # breakpoint()

    return binary_crossentropy(y_true[1], tf.expand_dims(y_pred, axis=-1), from_logits)

    # binary_crossentropy(tf.squeeze(y_true[1]), y_pred, from_logits)

    y_true = y_true[1]
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.expand_dims(y_pred, axis=-1)

    shape = y_true.get_shape()

    _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    # breakpoint()

    y1 = y_true * tf.math.log(y_pred)
    y2 = (1-y_true) * tf.math.log(1-y_pred)

    reduct_dim = range(len(shape))[1:]

    return -tf.reduce_mean(y1 + y2, axis=reduct_dim)

    modulator = tf.math.pow((1 - y_pred), gamma)

class TwoPathBCE(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_dice_loss """

    def __init__(self,
                 reduction,
                 name='2bce_manual',
                 from_logits=False,
                 **kwargs):
        super().__init__(
            two_path_bce,
            name=name,
            reduction=reduction,
            from_logits=from_logits
        )

def two_path_binary_dice(y_true, y_pred, from_logits=False, smooth=0.01):

    # breakpoint()

    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = y_true[1]
    shape = y_true.get_shape()
    reduction_dims = range(len(shape))[1:-1]

    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    union = tf.reduce_sum(y_true + y_pred, axis=reduction_dims)
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1.0 - tf.reduce_mean(dice, axis=-1, keepdims=True)


class TwoPathBinaryDice(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_dice_loss """

    def __init__(self,
                 reduction,
                 name='2bdice_manual',
                 from_logits=False,
                 **kwargs):
        super().__init__(
            two_path_binary_dice,
            name=name,
            reduction=reduction,
            from_logits=from_logits
        )



def surface_focal_loss(y_true, y_pred, gamma=2, class_weights=None):

    # Clip for numerical stability
    _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    y_pred = tf.expand_dims(y_pred, axis=-1)
    y_true = y_true[1]
    y_true = tf.cast(y_true, tf.float32)
    

    # y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    #

    shape = y_true.get_shape()
    reduction_dims = range(len(shape))[1:-1]

    # breakpoint()

    y1 = y_true * tf.math.log(y_pred)
    y2 = (1-y_true) * tf.math.log(1-y_pred)

    # modulator1 = tf.math.pow((1 - y_pred), gamma)
    #
    # modulator2 = tf.math.pow(y_pred, gamma)

    weight1 = 1.4
    weight2 = 0.7

    y1 *= weight1
    y2 *= weight2

    reduct_dim = range(len(shape))[1:]

    return -tf.reduce_mean(weight1 * y1 + weight2 * y2, axis=reduct_dim)


class SurfaceFocalLoss(LossFunctionWrapper):
    """
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, reduction, gamma=2,
                 class_weights=None, name="sparse_focal_loss"):
        super(SurfaceFocalLoss, self).__init__(
            surface_focal_loss,
            name=name,
            reduction=reduction,
            gamma=gamma,
            class_weights=class_weights
        )




def sparse_exponential_logarithmic_loss(y_true, y_pred, gamma_dice,
                                        gamma_cross, weight_dice,
                                        weight_cross):
    """
    TODO

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    # Clip for numerical stability
    _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    # Compute exp log dice
    intersect = 2 * tf.reduce_sum(y_true * y_pred, axis=reduction_dims) + 1
    union = tf.reduce_sum(y_true + y_pred, axis=reduction_dims) + 1
    exp_log_dice = tf.math.pow(-tf.math.log(intersect/union), gamma_dice)
    mean_exp_log_dice = tf.reduce_mean(exp_log_dice, axis=-1, keepdims=True)

    # Compute exp cross entropy
    entropy = tf.reduce_sum(y_true * -tf.math.log(y_pred), axis=-1, keepdims=True)
    exp_entropy = tf.reduce_mean(tf.math.pow(entropy, gamma_cross), axis=reduction_dims)

    # Compute output
    res = weight_dice*mean_exp_log_dice + weight_cross*exp_entropy
    return res


class SparseExponentialLogarithmicLoss(LossFunctionWrapper):
    """
    https://link.springer.com/content/pdf/10.1007%2F978-3-030-00931-1_70.pdf
    """
    def __init__(self, reduction, gamma_dice=0.3, gamma_cross=0.3,
                 weight_dice=1, weight_cross=1,
                 name="sparse_exponential_logarithmic_loss"):
        super(SparseExponentialLogarithmicLoss, self).__init__(
            sparse_exponential_logarithmic_loss,
            name=name,
            reduction=reduction,
            gamma_dice=gamma_dice,
            gamma_cross=gamma_cross,
            weight_dice=weight_dice,
            weight_cross=weight_cross
        )



def sparse_dice_and_ce_loss(y_true, y_pred, smooth=1):
    """
    TODO

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    dice_loss = sparse_dice_loss(y_true, y_pred)

    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    # Clip for numerical stability
    # _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
    # y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    # Compute the focal loss
    entropy = tf.math.log(y_pred)
    loss = -tf.reduce_sum(y_true * entropy, axis=-1, keepdims=True)
    return dice_loss + tf.reduce_mean(loss, axis=reduction_dims)


class SparseDiceAndCELoss(LossFunctionWrapper):
    """ tf reduction wrapper for sparse_dice_loss """
    def __init__(self,
                 reduction,
                 smooth=1,
                 name='sparse_dice_and_ce_loss',
                 **kwargs):
        super().__init__(
            sparse_dice_and_ce_loss,
            name=name,
            reduction=reduction,
            smooth=smooth
        )



def sparse_focal_loss(y_true, y_pred, gamma=2, class_weights=None):
    """
    TODO

    :param y_true:
    :param y_pred:
    :param smooth:
    :return:
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    # Clip for numerical stability
    _epsilon = _to_tensor(10e-8, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)

    if class_weights is None:
        class_weights = [1] * n_classes

    # Compute the focal loss
    entropy = tf.math.log(y_pred)
    modulator = tf.math.pow((1 - y_pred), gamma)
    loss = -tf.reduce_sum(class_weights * y_true * modulator * entropy, axis=-1, keepdims=True)
    return tf.reduce_mean(loss, axis=reduction_dims)


class SparseFocalLoss(LossFunctionWrapper):
    """
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, reduction, gamma=2,
                 class_weights=None, name="sparse_focal_loss"):
        super(SparseFocalLoss, self).__init__(
            sparse_focal_loss,
            name=name,
            reduction=reduction,
            gamma=gamma,
            class_weights=class_weights
        )


def sparse_generalized_dice_loss(y_true, y_pred, type_weight):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    """
    y_true, shape, n_classes = _get_shapes_and_one_hot(y_true, y_pred)
    reduction_dims = range(len(shape))[1:-1]

    ref_vol = tf.reduce_sum(y_true, axis=reduction_dims)
    intersect = tf.reduce_sum(y_true * y_pred, axis=reduction_dims)
    seg_vol = tf.reduce_sum(y_pred, axis=reduction_dims)

    if type_weight.lower() == 'square':
        weights = tf.math.reciprocal(tf.math.square(ref_vol))
    elif type_weight.lower() == 'simple':
        weights = tf.math.reciprocal(ref_vol)
    elif type_weight.lower() == 'uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))

    # Make array of new weight in which infinite values are replaced by
    # ones.
    new_weights = tf.where(tf.math.is_inf(weights),
                           tf.zeros_like(weights),
                           weights)

    # Set final weights as either original weights or highest observed
    # non-infinite weight
    weights = tf.where(tf.math.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)

    # calculate generalized dice score
    eps = 1e-6
    numerator = 2 * tf.multiply(weights, intersect)
    denom = tf.multiply(weights, seg_vol + ref_vol) + eps
    generalised_dice_score = numerator / denom
    return 1 - tf.reduce_mean(generalised_dice_score, axis=-1, keepdims=True)


class SparseGeneralizedDiceLoss(LossFunctionWrapper):
    """
    Based on implementation in NiftyNet at:

    http://niftynet.readthedocs.io/en/dev/_modules/niftynet/layer/
    loss_segmentation.html#generalised_dice_loss

    Class based to allow passing of parameters to the function at construction
    time in keras.
    """
    def __init__(self, reduction, type_weight="Square",
                 name='sparse_generalized_dice_loss'):
        super(SparseGeneralizedDiceLoss, self).__init__(
            sparse_generalized_dice_loss,
            name=name,
            reduction=reduction,
            type_weight=type_weight
        )


# Aliases
SparseExpLogDice = SparseExponentialLogarithmicLoss



class BinaryCrossentropyOneHot(LossFunctionWrapper):
  def __init__(self,
               from_logits=False,
               label_smoothing=0,
               reduction=losses_utils.ReductionV2.AUTO,
               name='binary_crossentropy_onehot'):

    super(BinaryCrossentropyOneHot, self).__init__(
        binary_crossentropy_onehot,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing)

    self.from_logits = from_logits


def binary_crossentropy_onehot(y_true, y_pred, from_logits=False, label_smoothing=0):

    y_pred = tf.math.reduce_sum(y_pred[..., 1:], axis=-1, keepdims=True)
    print(y_pred.shape)

    y_true = y_true > 0

    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)

    return binary_crossentropy(y_true, y_pred, from_logits, label_smoothing)


if __name__ == '__main__':
    import numpy as np

    # tf.compat.v1.enable_eager_execution()
    # import tensorflow.python.ops.numpy_ops.np_config

    # np_config.enable_numpy_behavior()

    y_true = tf.convert_to_tensor(np.ones((4, 2, 1)))

    y_true[:,:,0]

    y_pred = np.zeros((4, 2, 3))
    y_pred[0,1,1] = 0.9
    y_pred[0,1,0] = 0.1
    y_pred[0,0,0] = 1
    y_pred[1,0,0] = 1

    y_pred = tf.convert_to_tensor(y_pred)


    bce = BinaryCrossentropyOneHot(from_logits=False)

    bce(tf.reshape(y_true, (y_true.shape[0], -1)), tf.reshape(y_pred, (y_pred.shape[0], -1))).numpy()


    from tensorflow.keras.losses import BinaryCrossentropy

    bce = BinaryCrossentropy()

    bce(y_true, y_pred).numpy()

    res = binary_crossentropy(y_true, y_pred).numpy()

    y_true = tf.squeeze(y_true[1])

    y_true = tf.Variable(np.ones((2, 12, 1)))
    y_pred = tf.Variable(np.random.rand(2, 12, 1))

    shape = y_true.get_shape()
    y_true = tf.cast(y_true, tf.float64)

    y1 = y_true * tf.math.log(y_pred)
    y2 = (1-y_true) * tf.math.log(1-y_pred)

    # modulator = tf.math.pow((1 - y_pred), gamma)

    res2 = -tf.reduce_mean(y1 + y2, axis=range(len(shape))[1:])