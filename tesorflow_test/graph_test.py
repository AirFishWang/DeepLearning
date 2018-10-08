# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

g1 = tf.Graph()

with g1.as_default():
    a = tf.constant(np.ones((1, 2, 3)), name="a")
    with tf.variable_scope("foo"):
        # v = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
        v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

g2 = tf.Graph()
with g2.as_default():
    b = tf.constant(np.ones((2, 3, 4)), name="b")


with tf.Session(graph=g1) as sess:
    tf.initialize_all_variables().run()
    print sess.run(a)


with tf.Session(graph=g2) as sess:
    tf.initialize_all_variables().run()
    print sess.run(b)