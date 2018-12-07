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

target = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).reshape((2, 2, 3))
output = np.array([[1.0, 1.0, 2.0], [2.0, 1.0, 1.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]).reshape((2, 2, 3))


# target = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
# output = np.array([[0.25, 0.25, 0.5], [0.5, 0.25, 0.25]])

target = tf.constant(target, name="target")
output = tf.constant(output, name="output")

result = keras.backend.categorical_crossentropy(target, output)

with tf.Session() as sess:
    tf.initialize_all_variables().run()
    out = sess.run(result)
    print(out)
