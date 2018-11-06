# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     rnn_test
   Description :
   Author :        wangchun
   date：          18-11-6
-------------------------------------------------
   Change Activity:
                   18-11-6:
-------------------------------------------------
"""
import tensorflow as tf
import numpy as np


def basic_rnn_test():
    runn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units=128)
    print runn_cell.state_size


def lstm_test():
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=128)
    inputs = tf.placeholder(np.float32, shape=(32, 100)) # batch_size = 32
    h0 = lstm_cell.zero_state(32, np.float32)

    output, h1 = lstm_cell.call(inputs, h0)

    print "h1.h shape = ".format(h1.h)
    print "h1.c shape = ".format(h1.c)


if __name__ == "__main__":
    basic_rnn_test()
    lstm_test()