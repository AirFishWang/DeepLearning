# -*- coding: utf-8 -*-
import tensorflow as tf


def tensor_shape_test():
    a = tf.Variable(451, tf.int16)
    print "a.shape = {}".format(a.shape)

    b = tf.Variable([2, 3, 5, 7, 11], tf.int32)
    print "b.shape = {}".format(b.shape)

    c = tf.Variable([[4], [9], [16], [25]], tf.int32)
    print "c.shape = {}".format(c.shape)

    d = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
    print "d.shape = {}".format(d.shape)

    x = tf.placeholder(tf.float32, shape=(None, 1080, 1920, 3))  # BHWC  (channel last)
    print "x.shape = {}".format(x.shape)


if __name__ == "__main__":
    tensor_shape_test()