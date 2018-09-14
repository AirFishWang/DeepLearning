#! -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

with tf.Graph().as_default():
    output_graph_def = tf.GraphDef()
    output_graph_path = './model/expert-graph.pb'

    with open(output_graph_path, 'rb') as f:
        output_graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(output_graph_def, name="")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")
        y_conv_2 = sess.run(output, feed_dict={input: mnist.test.images})
        y_2 = mnist.test.labels
        correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y_2, 1))
        accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
        print "check accuracy %g" % sess.run(accuracy_2)