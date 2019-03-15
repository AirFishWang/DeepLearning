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
    #  此focal loss函数只适用于二分类问题和单分类问题(当然单分类和二分类本质上属于一个问题)
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

    cls_loss = focal_weight * tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=classification)

    return tf.reduce_sum(cls_loss)


# target = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).reshape((4, 3))
# output = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).reshape((4, 3))
# output2 = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).reshape((4, 3))


target = np.array([1.0, 0.0])
out = np.array([4.0, 2.0])
out2 = np.array([3.0, 2.0])

target = tf.constant(target, name="target")
out = tf.constant(out, name="out")

softmax_out = tf.nn.softmax(out)
sigmoid_out = tf.nn.sigmoid(out)
sigmoid_out2 = tf.nn.sigmoid(out2)

ce = keras.backend.categorical_crossentropy(target, softmax_out)
sc = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=out)


fl = focal(target, sigmoid_out, alpha=1, gamma=2)
f2 = focal(target, sigmoid_out2, alpha=1, gamma=2)


with tf.Session() as sess:
    tf.initialize_all_variables().run()
    print "softmax_out = {}".format(sess.run(softmax_out))
    print "sigmoid_out = {}".format(sess.run(sigmoid_out))
    print "categorical_crossentropy = {}".format(sess.run(ce))
    print "sigmoid_cross_entropy_with_logits = {}".format(sess.run(sc))

    print "f1 = {}".format(sess.run(fl))
    print "f2 = {}".format(sess.run(f2))
