# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     my_loss
   Description :
   Author :        wangchun
   date：          19-3-16
-------------------------------------------------
   Change Activity:
                   19-3-16:
-------------------------------------------------
"""
import numpy as np
import tensorflow as tf


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


def my_cls_ohem(cls_prob, label, num_keep_radio=0.7):
    """
    reference MTCNN: https://github.com/AITTSMD/MTCNN-Tensorflow/blob/master/train_models/mtcnn_model.py
    :param cls_prob:  a output vector of softmax
    :param label:     ground truth(one-hot vector)
    :return:
    """
    label = tf.cast(tf.argmax(label, axis=1), dtype=tf.float32)

    zeros = tf.zeros_like(label)
    #label=-1 --> label=0net_factory

    #pos -> 1, neg -> 0, others -> 0
    label_filter_invalid = tf.where(tf.less(label,0), zeros, label)
    num_cls_prob = tf.size(cls_prob)  # 返回tensor中元素的数量，(5,2)的tensor 输出10
    cls_prob_reshape = tf.reshape(cls_prob,[num_cls_prob,-1])   # (5,2) -> (10,1)
    label_int = tf.cast(label_filter_invalid,tf.int32)          # 数据类型转换 转成tf.int32
    # get the number of rows of class_prob
    num_row = tf.to_int32(cls_prob.get_shape()[0])
    #row = [0,2,4.....]
    row = tf.range(num_row)*2
    indices_ = row + label_int
    label_prob = tf.squeeze(tf.gather(cls_prob_reshape, indices_))   # tf.gather按照indices_索引，从cls_prob_reshape中取元素
    loss = -tf.log(label_prob+1e-10)
    zeros = tf.zeros_like(label_prob, dtype=tf.float32)
    ones = tf.ones_like(label_prob,dtype=tf.float32)
    # set pos and neg to be 1, rest to be 0
    valid_inds = tf.where(label < zeros,zeros,ones)
    # get the number of POS and NEG examples
    num_valid = tf.reduce_sum(valid_inds)

    keep_num = tf.cast(num_valid*num_keep_radio,dtype=tf.int32)
    #FILTER OUT PART AND LANDMARK DATA
    loss = loss * tf.cast(valid_inds, dtype=tf.float64)
    topk_loss,_ = tf.nn.top_k(loss, k=keep_num)
    # return loss, tf.reduce_mean(topk_loss)  # debug
    return tf.reduce_mean(topk_loss)


def ohem_loss(target, output, num_keep_radio=0.7):
    """
    :param target: one-hot vector
    :param output: softmax output
    :param num_keep_radio:
    :return:
    """
    output /= tf.reduce_sum(output, len(output.get_shape()) - 1, True)  # reference keras.backend.categorical_crossentropy
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(1e-7, dtype=output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)

    loss = tf.reduce_sum(- target * tf.log(output), axis=-1)
    batch_size = (target.get_shape()[0]).value
    topk_loss, _ = tf.nn.top_k(loss, k=int(batch_size*num_keep_radio))
    # return loss, tf.reduce_mean(topk_loss)  # debug
    return tf.reduce_mean(topk_loss)


if __name__ == "__main__":
    gt_label = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    y        = np.array([[0.5, 0.5], [0.8, 0.2], [0.1, 0.9], [0.1, 0.9], [0.4, 0.6]])

    gt_label = tf.constant(gt_label, name="gt_label")
    y = tf.constant(y, name="y")
    loss = my_cls_ohem(y, gt_label)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print "encode label = {}".format(sess.run(gt_label))
        topk_mean_loss = sess.run(loss)
        print "loss = {}".format(loss)
        print "topk_mean_loss = {}".format(topk_mean_loss)
        print sess.run(ohem_loss(gt_label, y))

