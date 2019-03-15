# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cross_entropy
   Description :
   Author :        wangchun
   date：          18-12-6
-------------------------------------------------
   Change Activity:
                   18-12-6:
-------------------------------------------------
"""
import keras
import tensorflow as tf
import numpy as np


def focal(y_true, y_pred, alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    classification = y_pred
    labels = y_true

    # compute the focal loss
    alpha_factor = tf.ones_like(labels) * alpha  # tf.ones_like(x) : 创建形状和x相同,所有元素为1的张量
    alpha_factor = tf.where(tf.equal(labels, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(tf.equal(labels, 1), 1 - classification, classification)
    focal_weight = alpha_factor * focal_weight ** gamma

    cls_loss = focal_weight * keras.backend.categorical_crossentropy(labels, classification)
    return tf.reduce_sum(cls_loss)


target = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).reshape((4, 3))
output = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).reshape((4, 3))


# target = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
# output = np.array([[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]])

target = np.array([1.0, 0.0, 0.0])
output = np.array([0.1, 0.4, 0.5])
output2 = np.array([0.2, 0.4, 0.4])

target = tf.constant(target, name="target")
output = tf.constant(output, name="output")
output2 = tf.constant(output2, name="output2")

ce = keras.backend.categorical_crossentropy(target, output)

fl = focal(target, output, alpha=1, gamma=2)
f2 = focal(target, output2, alpha=1, gamma=2)


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print sess.run(ce)
    print sess.run(fl)
    print sess.run(f2)

