# -*- coding: utf-8 -*-
"""
tensorboard --logdir=./log
然后打开浏览器，输入地址： localhost:6006   (6006是默认端口)

"""
import tensorflow as tf

input1 = tf.constant([1.0, 2.0, 3.0], name="input1")
input2 = tf.Variable(tf.random_uniform([3]), name="input2")

output = tf.add_n([input1, input2], name="add")

writer = tf.summary.FileWriter('./log', tf.get_default_graph())

writer.close()