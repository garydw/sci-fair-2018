import tensorflow as tf
import numpy as np

WEIGHT_DECAY = 1e-4
BN_MOMENTUM = 0.997
EPSILON = 1e-4


def get_v(name, shape, initializer=tf.variance_scaling_initializer, decay=None, trainable=True, dtype='float'):
    regularizer = None
    if decay:
        regularizer = tf.contrib.layers.l2_regularizer(decay)
    return tf.get_variable(name, shape=shape, initializer=initializer, regularizer=regularizer, trainable=trainable, dtype=dtype)

def conv(x, out, ksize=3, stride=1):
    weights = get_v('weights', [ksize, ksize, x.get_shape()[-1], out], decay=WEIGHT_DECAY)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

def fc(x, units, scope):
    with tf.variable_scope(scope):
        weights = get_v('weights', [x.get_shape()[1], units], decay=WEIGHT_DECAY)
        biases = get_v('biases', [units], initializer=tf.zeros_initializer)
        x = tf.nn.xw_plus_b(x, weights, biases)
    return x

def bn(x, training):
    return tf.layers.batch_normalization(x, momentum=BN_MOMENTUM, epsilon=EPSILON, training=training)

def activation(x, typ='relu'):
    if typ == 'relu':
        return tf.nn.relu(x)

def block(x, training, bottleneck=False, downsample=False):
    identity = x
    stride = 2 if downsample else 1
    out = x.get_shape()[-1] * 2 if downsample else x.get_shape()[-1]
    if bottleneck:
        internal = x.get_shape()[-1] / 2 if downsample else x.get_shape()[-1] / 4
        with tf.variable_scope('bottle_1'):
            x = conv(x, internal, ksize=1, stride=stride) 
            x = bn(x, training)
            x = activation(x)
        with tf.variable_scope('bottle_2'):
            x = conv(x, internal)
            x = bn(x, training)
            x = activation(x)
        with tf.variable_scope('bottle_3'):
            x = conv(x, internal * 4, ksize=1)
            x = bn(x, training)
    else:
        with tf.variable_scope('reg_1'):
            x = conv(x, out, stride=stride)
            x = bn(x, training)
            x = activation(x)
        with tf.variable_scope('reg_2'):
            x = conv(x, out)
            x = bn(x, training)
    with tf.variable_scope('identity'):
        if downsample:
            in_channels = identity.get_shape()[-1]
            identity = tf.nn.avg_pool(identity, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
            channels = int((int(x.get_shape()[-1]) - int(in_channels)) / 2)
            padding = [[0, 0], [0, 0], [0, 0], [channels, channels]]
            identity = tf.pad(identity, padding)
        x = activation(x + identity)
    return x

def group(x, nblock, scope, bottleneck=False, training=True, downsample=True):
    with tf.variable_scope(scope):
        for i in range(nblock):
            if downsample:
                downsample = True if i == 0 else False
            else:
                downsample = False if i > 0 else True
            with tf.variable_scope('block{}'.format(i + 1)):
                x = block(x, training, bottleneck=bottleneck, downsample=downsample)
    return x

def flatten(x):
    a, b, c = x.get_shape()[1:]
    x = tf.reshape(x, [-1, a * b * c])
    return x

def max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], 'SAME')

def avg_pool(x, ksize=3, stride=2):
    return tf.nn.avg_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], 'SAME')