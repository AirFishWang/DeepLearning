# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

# a = tf.constant([1.0, 2.0, 3.0], name="a")
# b = tf.constant([1.0, 2.0, 3.0], name="b")

a = tf.constant(np.ones((1, 2, 3)), name="a")
b = tf.constant(np.ones((1, 2, 3)), name="b")

add_result = a + b
sub_result = a - b

with tf.Session() as sess:
    output = sess.run([add_result, sub_result])
    print output[0]
    print output[1]