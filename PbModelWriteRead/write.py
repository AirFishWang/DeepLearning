# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import shutil
import os.path

export_dir = './model/'
if os.path.exists(export_dir):
    shutil.rmtree(export_dir)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


mnist = input_data.read_data_sets("../data/MNIST_data/", one_hot=True)

with tf.Graph().as_default():
    ## 变量占位符定义
    x = tf.placeholder("float", shape=[None, 784])
    y_ = tf.placeholder("float", shape=[None, 10])

    ## 定义网络结构
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    #
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    #
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    #
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    #
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    #
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    ## 定义损失及优化器
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        ## 初始化变量
        sess.run(tf.global_variables_initializer())
        for i in range(201):
            batch = mnist.train.next_batch(50)
            if i % 100 == 0:
                ## 验证阶段dropout比率为1
                train_accuracy = sess.run(accuracy, feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
                print "step %d, training accuracy %g" % (i, train_accuracy)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        print('test accuracy %g' % sess.run(accuracy,
                                            feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

        ## 将网络中的权值变量取出来
        _W_conv1 = sess.run(W_conv1)
        _b_conv1 = sess.run(b_conv1)
        _W_conv2 = sess.run(W_conv2)
        _b_conv2 = sess.run(b_conv2)
        _W_fc1 = sess.run(W_fc1)
        _b_fc1 = sess.run(b_fc1)
        _W_fc2 = sess.run(W_fc2)
        _b_fc2 = sess.run(b_fc2)

## 创建另外一个图，验证权值的正确性并save model
with tf.Graph().as_default():
    ## 定义变量占位符
    x_2 = tf.placeholder("float", shape=[None, 784], name="input")
    y_2 = tf.placeholder("float", [None, 10])

    ## 网络的权重用上一个图中已经学习好的对应值
    W_conv1_2 = tf.constant(_W_conv1, name="constant_W_conv1")
    b_conv1_2 = tf.constant(_b_conv1, name="constant_b_conv1")
    x_image_2 = tf.reshape(x_2, [-1, 28, 28, 1])
    h_conv1_2 = tf.nn.relu(conv2d(x_image_2, W_conv1_2) + b_conv1_2)
    h_pool1_2 = max_pool_2x2(h_conv1_2)
    #
    W_conv2_2 = tf.constant(_W_conv2, name="constant_W_conv2")
    b_conv2_2 = tf.constant(_b_conv2, name="constant_b_conv2")
    h_conv2_2 = tf.nn.relu(conv2d(h_pool1_2, W_conv2_2) + b_conv2_2)
    h_pool2_2 = max_pool_2x2(h_conv2_2)
    #
    W_fc1_2 = tf.constant(_W_fc1, name="constant_W_fc1")
    b_fc1_2 = tf.constant(_b_fc1, name="constant_b_fc1")
    h_pool2_flat_2 = tf.reshape(h_pool2_2, [-1, 7 * 7 * 64])
    h_fc1_2 = tf.nn.relu(tf.matmul(h_pool2_flat_2, W_fc1_2) + b_fc1_2)
    #
    # DropOut is skipped for exported graph.
    ## 由于是验证过程，所以dropout层去掉，也相当于keep_prob为1
    #
    W_fc2_2 = tf.constant(_W_fc2, name="constant_W_fc2")
    b_fc2_2 = tf.constant(_b_fc2, name="constant_b_fc2")
    #
    y_conv_2 = tf.nn.softmax(tf.matmul(h_fc1_2, W_fc2_2) + b_fc2_2, name="output")

    with tf.Session() as sess_2:
        sess_2.run(tf.global_variables_initializer())
        tf.train.write_graph(sess_2.graph_def, export_dir, 'expert-graph.pb', as_text=False)  #
        correct_prediction_2 = tf.equal(tf.argmax(y_conv_2, 1), tf.argmax(y_2, 1))
        accuracy_2 = tf.reduce_mean(tf.cast(correct_prediction_2, "float"))
        print('check accuracy %g' % sess_2.run(accuracy_2, feed_dict={x_2: mnist.test.images, y_2: mnist.test.labels}))