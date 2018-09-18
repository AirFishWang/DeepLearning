#!/usr/bin/python
# -*- coding: utf-8 -*-

#
# Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.examples.tutorials.mnist import input_data
    from tensorflow.python.tools import freeze_graph
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have tensorflow installed.
https://www.tensorflow.org/install/""".format(err))

try:
    import uff
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the UFF Toolkit installed.""".format(err))

path = os.path.dirname(os.path.realpath(__file__))

STARTER_LEARNING_RATE = 1e-4
BATCH_SIZE = 10
NUM_CLASSES = 10
MAX_STEPS = 5000
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE ** 2
OUTPUT_NAMES = ["fc2/Relu"]
UFF_OUTPUT_FILENAME = path + "/lenet5.uff"

MNIST_DATASETS = input_data.read_data_sets('mnist/input_data')


def WeightsVariable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.1, name='weights'))


def BiasVariable(shape):
    return tf.Variable(tf.constant(0.1, shape=shape, name='biases'))


def Conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    filter_size = W.get_shape().as_list()
    pad_size = filter_size[0]//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    x = tf.pad(x, pad_mat)
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def MaxPool2x2(x, k=2):
    # MaxPool2D wrapper
    pad_size = k//2
    pad_mat = np.array([[0,0],[pad_size,pad_size],[pad_size,pad_size],[0,0]])
    #x = tf.pad(x, pad_mat)
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def network(images_reshape):
    # Reshape
    #with tf.name_scope('reshape'):
    #    images_reshape = tf.reshape(images, [-1, 28, 28, 1])

    # Convolution 1
    with tf.name_scope('conv1'):
        weights = WeightsVariable([5,5,1,32])
        biases = BiasVariable([32])
        conv1 = tf.nn.relu(Conv2d(images_reshape, weights, biases))
        pool1 = MaxPool2x2(conv1)

    # Convolution 2
    with tf.name_scope('conv2'):
        weights = WeightsVariable([5,5,32,64])
        biases = BiasVariable([64])
        conv2 = tf.nn.relu(Conv2d(pool1, weights, biases))
        pool2 = MaxPool2x2(conv2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # Fully Connected 1
    with tf.name_scope('fc1'):
        weights = WeightsVariable([7 * 7 * 64, 1024])
        biases = BiasVariable([1024])
        fc1 = tf.nn.relu(tf.matmul(pool2_flat, weights) + biases)

    # Fully Connected 2
    with tf.name_scope('fc2'):
        weights = WeightsVariable([1024, 10])
        biases = BiasVariable([10])
        fc2 = tf.reshape(tf.matmul(fc1,weights) + biases, shape=[-1,10], name='Relu')

    return fc2


def loss_metrics(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='softmax')
    return tf.reduce_mean(cross_entropy, name='softmax_mean')


def training(loss):
    tf.summary.scalar('loss', loss)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    learning_rate = tf.train.exponential_decay(STARTER_LEARNING_RATE, global_step, 100000, 0.75, staircase=True)
    tf.summary.scalar('learning rate', learning_rate)
    optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):
    images_placeholder = tf.placeholder(tf.float32, shape=(None, 28, 28, 1)) # Tensor("Placeholder:0", shape=(?, 28, 28, 1), dtype=float32)
    labels_placeholder = tf.placeholder(tf.int32, shape=(None)) # Tensor("Placeholder_1:0", dtype=int32)
    return images_placeholder, labels_placeholder


def fill_feed_dict(data_set, images_pl, labels_pl):
    images_feed, labels_feed = data_set.next_batch(BATCH_SIZE)
    feed_dict = {
        images_pl: np.reshape(images_feed, (-1,28,28,1)),
        labels_pl: labels_feed,
    }
    return feed_dict


def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            summary):

    true_count = 0
    steps_per_epoch = data_set.num_examples // BATCH_SIZE
    num_examples = steps_per_epoch * BATCH_SIZE
    for step in range(steps_per_epoch):
        feed_dict = fill_feed_dict(data_set,
                                   images_placeholder,
                                   labels_placeholder)
        log, correctness = sess.run([summary, eval_correct], feed_dict=feed_dict)
        true_count += correctness
    precision = float(true_count) / num_examples
    tf.summary.scalar('precision', tf.constant(precision))
    print('Num examples %d, Num Correct: %d Precision @ 1: %0.04f' % (num_examples, true_count, precision))
    return log


def run_training(data_sets):
    with tf.Graph().as_default():
        images_placeholder, labels_placeholder = placeholder_inputs(BATCH_SIZE)
        logits = network(images_placeholder)
        loss = loss_metrics(logits, labels_placeholder)
        train_op = training(loss)
        eval_correct = evaluation(logits, labels_placeholder)
        summary = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        summary_writer = tf.summary.FileWriter("mnist/log", graph=tf.get_default_graph())
        test_writer = tf.summary.FileWriter("mnist/log/validation",  graph=tf.get_default_graph())
        sess.run(init)
        for step in range(MAX_STEPS):
            start_time = time.time()
            feed_dict = fill_feed_dict(data_sets.train,
                                       images_placeholder,
                                       labels_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            duration = time.time() - start_time
            if step % 100 == 0:
                print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()
            if (step + 1) % 1000 == 0 or (step + 1) == MAX_STEPS:
                checkpoint_file = os.path.join("mnist/log", "model.ckpt")
                saver.save(sess, checkpoint_file, global_step=step)
                print('Validation Data Eval:')
                log = do_eval(sess,
                              eval_correct,
                              images_placeholder,
                              labels_placeholder,
                              data_sets.validation,
                              summary)
                test_writer.add_summary(log, step)
        #return sess

        graphdef = tf.get_default_graph().as_graph_def()
        frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, OUTPUT_NAMES)
        return tf.graph_util.remove_training_nodes(frozen_graph)


def learn():
    return run_training(MNIST_DATASETS)


def get_testcase():
    return MNIST_DATASETS.test.next_batch(1)


def load_meta_model_eval():
    meta_file = os.path.join("mnist/log", "model.ckpt-4999.meta")
    weight_file = os.path.join("mnist/log", "model.ckpt-4999")
    with tf.Graph().as_default():
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(meta_file)
            saver.restore(sess, weight_file)

            eval_correct = tf.get_default_graph().get_tensor_by_name("Sum:0")
            images_placeholder = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
            labels_placeholder = tf.get_default_graph().get_tensor_by_name("Placeholder_1:0")
            summary = tf.get_default_graph().get_tensor_by_name("Merge/MergeSummary:0")
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    MNIST_DATASETS.validation,
                    summary)

            # write to pb file (pb means protobuff)
            # 注意： 在tensorRT的demo， output_node_names其实只需要一个fc2/Relu, 这里改为多个是为了测试后面的do_eval函数
            save_way = 1
            if save_way == 1:
                # 参考： https://www.cnblogs.com/Time-LCJ/p/8449646.html
                # 此方法等效于用tensorflow的freeze_graph.py 脚本工具来固化
                tf.train.write_graph(sess.graph_def, 'mnist/log', '4999.pb')
                freeze_graph.freeze_graph(input_graph="mnist/log/4999.pb",
                                          input_saver='',
                                          input_binary=False,
                                          input_checkpoint=weight_file,
                                          output_node_names="fc2/Relu, Sum, Placeholder_1, Merge/MergeSummary",  # 逗号分隔多个节点
                                          restore_op_name="save/restore_all",
                                          filename_tensor_name="save/Const:0",
                                          output_graph="mnist/log/4999.pb",
                                          clear_devices=True,
                                          initializer_nodes='')
            elif save_way == 2:
                # tf.get_default_graph(): 获得图
                # tf.get_default_graph().as_graph_def(): 获得序列化的图, 序列化的图可以用于tf.import_graph_def函数
                graph_def = tf.get_default_graph().as_graph_def()
                # frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graphdef, OUTPUT_NAMES)
                frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['fc2/Relu', "Sum", "Placeholder_1", "Merge/MergeSummary"])
                frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)  # 移除train节点可以更加简化模型
                with tf.gfile.GFile("mnist/log/4999.pb", "wb") as f:
                    f.write(frozen_graph.SerializeToString())
                print("%d ops in the final graph in function load_meta_model_eval" % (len(frozen_graph.node)))

                # type(tf.get_default_graph().get_operations()) = list
                for index, op in enumerate(tf.get_default_graph().get_operations()):
                    print("{} op.name in function load_meta_model_eval: {}".format(index+1, op.name))
            else:
                print("Please select one way to save pb model file")


def load_pb_model_eval():
    # exit()
    pb_file = "mnist/log/4999.pb"

    with tf.gfile.GFile(pb_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        eval_correct, images_placeholder, labels_placeholder, summary = [None, None, None, None]

        load_way = 2
        if load_way == 1:
            # 方式1
            tf.import_graph_def(graph_def)  # 未指定参数name, name会取默认值"import", 会将图里的张量名前面加上"import/"

            for op in graph.get_operations():
                print("in function load_pb_model_eval: {}".format(op.name))

            eval_correct = graph.get_tensor_by_name("import/Sum:0")
            images_placeholder = graph.get_tensor_by_name("import/Placeholder:0")
            labels_placeholder = graph.get_tensor_by_name("import/Placeholder_1:0")
            summary = graph.get_tensor_by_name("import/Merge/MergeSummary:0")
        elif load_way == 2:
            # 方式2
            tf.import_graph_def(graph_def, name="")

            for op in graph.get_operations():
                print("in function load_pb_model_eval: {}".format(op.name))

            eval_correct = graph.get_tensor_by_name("Sum:0")
            images_placeholder = graph.get_tensor_by_name("Placeholder:0")
            labels_placeholder = graph.get_tensor_by_name("Placeholder_1:0")
            summary = graph.get_tensor_by_name("Merge/MergeSummary:0")
        elif load_way == 3:
            # 方式3
            eval_correct, images_placeholder, labels_placeholder, summary = \
                tf.import_graph_def(graph_def, return_elements=["Sum:0", "Placeholder:0", "Placeholder_1:0", "Merge/MergeSummary:0"])
        else:
            print("Please select one way to load pb model file")

        with tf.Session(graph=graph) as sess:
            do_eval(sess,
                    eval_correct,
                    images_placeholder,
                    labels_placeholder,
                    MNIST_DATASETS.validation,
                    summary)


if __name__ == "__main__":
    # frozen_graph = run_training(MNIST_DATASETS)
    # uff.from_tensorflow(graphdef=frozen_graph,
    #                     output_filename=UFF_OUTPUT_FILENAME,
    #                     output_nodes=OUTPUT_NAMES,
    #                     text=True)

    load_meta_model_eval()
    load_pb_model_eval()


