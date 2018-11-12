# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def tensor_shape_test():
    a = tf.Variable(451, tf.int16)
    print "a.shape = {}".format(a.shape)

    b = tf.Variable([2, 3, 5, 7, 11], tf.int32)
    print "b.shape = {}".format(b.shape)

    c = tf.Variable([[4], [9], [16], [25]], tf.int32)
    print "c.shape = {}".format(c.shape)

    d = tf.Variable([ [4, 9, 0], [16, 25, 1]], tf.int32)
    print "d.shape = {}".format(d.shape)

    x = tf.placeholder(tf.float32, shape=(None, 1080, 1920, 3))  # BHWC  (channel last)
    print "x.shape = {}".format(x.shape)

    concat_d = tf.concat([d, d], 0)
    print "concat_d.shape = {}".format(concat_d.shape)

    e = tf.placeholder(tf.float32, shape=(16, 20, 512), name='e')
    e_split = tf.split(e, 16, 0)
    for x in e_split:
        print x.name, x.shape

    e_split = [tf.squeeze(x, squeeze_dims=[0]) for x in e_split]
    for x in e_split:
        print x.name, x.shape

    # input = np.ones((14, 1, 512), dtype=np.float32)
    # with tf.Session() as sess:
    #     result = sess.run(e_split, feed_dict={e.name : input})
    #     for x in result:
    #         print x.shape

if __name__ == "__main__":
    tensor_shape_test()