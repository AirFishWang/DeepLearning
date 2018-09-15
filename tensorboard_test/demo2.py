# -*- coding: utf-8 -*-
"""
tensorboard --logdir=./log
然后打开浏览器，输入地址： localhost:6006   (6006是默认端口)

tips: 双击图上的节点或名称空间会看到详细信息
"""

import tensorflow as tf


def load_lenet5_meta_model():
    meta_file = "/home/wangchun/Desktop/polyp/DeepLearning/tensorrt_site_package/examples/tf_to_trt/mnist/log/model.ckpt-4999.meta"
    weight_file = "/home/wangchun/Desktop/polyp/DeepLearning/tensorrt_site_package/examples/tf_to_trt/mnist/log/model.ckpt-4999"
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, weight_file)

            writer = tf.summary.FileWriter('./log', tf.get_default_graph())
            writer.close()


def load_ocr_meta_model():
    meta_file = '/home/wangchun/Desktop/Attention-OCR/Attention-OCR-v2-debug/snapshots/translate.ckpt-850000.meta'
    weight_file = '/home/wangchun/Desktop/Attention-OCR/Attention-OCR-v2-debug/snapshots/translate.ckpt-850000'
    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, weight_file)

            writer = tf.summary.FileWriter('./log', tf.get_default_graph())
            writer.close()


if __name__ == "__main__":
    # load_lenet5_meta_model()
    load_ocr_meta_model()