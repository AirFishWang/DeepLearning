# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     model_test
   Description :
   Author :        wangchun
   date：          19-3-2
-------------------------------------------------
   Change Activity:
                   19-3-2:
-------------------------------------------------
"""
import math
import numpy as np
import tensorflow as tf


class PriorProbability(tf.keras.initializers.Initializer):
    """ Apply a prior probability to the weights.
    """

    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {
            'probability': self.probability
        }

    def __call__(self, shape, dtype=None, partition_info=None):
        # set bias to -log((1 - p)/p) for foreground
        dtype = None
        result = np.ones(shape, dtype=dtype) * -math.log((1 - self.probability) / self.probability)

        return result


def default_regression_model(num_anchors, pyramid_feature_size=256, regression_feature_size=256, name='regression_submodel'):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer': 'zeros'
    }

    inputs = tf.keras.layers.Input(shape=(512, 512, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = tf.keras.layers.Conv2D(
            filters=regression_feature_size,
            activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)

    outputs = tf.keras.layers.Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = tf.keras.layers.Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)


def default_classification_model(
    num_classes,
    num_anchors,
    pyramid_feature_size=256,
    prior_probability=0.01,
    classification_feature_size=256,
    name='classification_submodel'
):
    """ Creates the default regression submodel.

    Args
        num_classes                 : Number of classes to predict a score for at each feature level.
        num_anchors                 : Number of anchors to predict classification scores for at each feature level.
        pyramid_feature_size        : The number of filters to expect from the feature pyramid levels.
        classification_feature_size : The number of filters to use in the layers in the classification submodel.
        name                        : The name of the submodel.

    Returns
        A keras.models.Model that predicts classes for each anchor.
    """
    options = {
        'kernel_size' : 3,
        'strides'     : 1,
        'padding'     : 'same',
    }

    inputs = tf.keras.layers.Input(shape=(512, 512, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = tf.keras.layers.Conv2D(
            filters=classification_feature_size,
            activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros',
            **options
        )(outputs)

    outputs = tf.keras.layers.Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=tf.keras.initializers.Zeros(),
        bias_initializer=PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = tf.keras.layers.Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = tf.keras.layers.Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name=name)

if __name__ == "__main__":
    model = default_regression_model(9)
    model = default_classification_model(1, 9)