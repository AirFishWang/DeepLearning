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


with tf.Session(graph=g1) as sess1:
    tf.initialize_all_variables().run()
    print sess1.run(a)


with tf.Session(graph=g2) as sess2:
    tf.initialize_all_variables().run()
    print sess2.run(b)


g3 = tf.Graph()
with g3.as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess3 = tf.Session(config=config)