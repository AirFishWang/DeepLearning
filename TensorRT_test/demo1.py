# -*- coding: utf-8 -*-
"""
注意事项：
(1) tensorflow程序在未指定会话GPU显存使用策略时, 会默认占用所有显存, 如果在一个程序中，先生成frozen_graph,然后直接转换UFF,此时由于没有可用显存, 会转换失败
    解决方法：一种解决方法是指定会话使用显存的方式，不让其占满； 另一种是跑两个程序,第一个程序将frozen_graph保存为pb模型, 然后用另一个程序加载模型并转换模型
(2) "import pycuda.autoinit" 这条语句是必须的， 否则会报 pycuda._driver.LogicError

(3) 关于register_iutput的shape问题：  register_input的形状为CHW, 即通道, 高, 宽, 下面讨论标量和一般向量的情况
    其实不管input_data是什么形状, 只要input_data可以正确被reshape为register_output指定的形状, 就可以
    例如： i: input_data是标量, 即一个数， 那么register_input的形状必须是(1,1,1), 因为一个数只能被reshape到(1,1,1)
        ii: input_data是一维的向量， 长度为n, 那么register_input的形状可以是(1,n,1)或者(1,1,n), 不能是(n,1,1)的原因是第一维代表C
        iii: 在tensorRT的官方Demo(tf_to_trt.py)中, register_input的指定shape是(1,28,28),但是input_data的shape却是(784,),但是程序不会出错, 原因是因为
            (784,)可以正确reshape到(1, 28, 28), 且第一维代表channel。经过测试即使input_data的shape即使是(28,7,4)这种奇怪的形状, 程序一样可以得到正确的结果

"""
import time
import os
import numpy as np
import tensorflow as tf
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser
import pycuda.driver as cuda
import pycuda.autoinit

MAX_WORKSPACE = 1 << 30
MAX_BATCHSIZE = 1
G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)


def get_frozen_graph():
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    with tf.Graph().as_default() as graph:
        # a = tf.placeholder(tf.float32, shape=(None, 2, 4, 1), name="a")  # BHWC
        # b = tf.nn.avg_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name="b")
        # c = tf.reshape(a, shape=[-1, 8], name="c")

        # a = tf.placeholder(tf.float32, shape=(), name="a")
        a = tf.placeholder(tf.float32, shape=(2,), name="a")
        b = tf.add(a, a, name="b")
        c = tf.subtract(a, a, name="c")

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # print sess.run(b, feed_dict={a: np.ones((1, 2, 4, 1))})                  # shape = (None, 2, 4, 1)
            # print sess.run(b, feed_dict={a: np.array(8).astype(np.float32)})         # shape = ()
            print sess.run(b, feed_dict={a: np.array([1.0, 2.0])})                      # shape = (2, )
            writer = tf.summary.FileWriter('./log', tf.get_default_graph())
            writer.close()

            graph_def = graph.as_graph_def()
            frozen_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, ["b", "c"])
            print "get_frozen_graph() finished"
            with tf.gfile.GFile("demo1.pb", "wb") as f:
                f.write(frozen_graph.SerializeToString())
            return frozen_graph
            # return tf.graph_util.remove_training_nodes(frozen_graph)


def test_frozen_graph(frozen_graph):
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    with tf.Graph().as_default() as graph:
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(frozen_graph.SerializeToString())
        tf.import_graph_def(graph_def, name="")

        a = graph.get_tensor_by_name("a:0")
        b = graph.get_tensor_by_name("b:0")

        with tf.Session() as sess:
            print sess.run(b, feed_dict={a: np.ones((1, 2, 4, 1))})


def tensorrt_infer(tf_model = None):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    batch_size = 1
    # input_data = np.ones(8).astype(np.float32)             # shape = (None, 2, 4, 1)
    # input_data = np.array(8).astype(np.float32)            # shape = ()
    input_data = np.array([1.0, 2.0]).astype(np.float32)     # shape = (2, )


    uff_model = uff.from_tensorflow(tf_model, ["b", "c"])
    # uff_model = uff.from_tensorflow_frozen_model("demo1.pb", ["b"])
    # Convert Tensorflow model to TensorRT model

    parser = uffparser.create_uff_parser()
    # parser.register_input("a", (1, 2, 4), 0)
    # parser.register_input("a", (1, 1, 1), 0)
    parser.register_input("a", (1, 1, 2), 0)
    parser.register_output("b")
    parser.register_output("c")
    engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, MAX_BATCHSIZE, MAX_WORKSPACE)
    assert (engine)
    parser.destroy()
    context = engine.create_execution_context()

    dims_a = engine.get_binding_dimensions(0).to_DimsCHW()
    dims_b = engine.get_binding_dimensions(1).to_DimsCHW()
    dims_c = engine.get_binding_dimensions(2).to_DimsCHW()

    # load engine
    engine = context.get_engine()
    assert (engine.get_nb_bindings() == 3)          # engine.get_nb_bindings() = register_input + register_output

    # Allocate pagelocked memory
    output_b = cuda.pagelocked_empty(dims_b.C() * dims_b.H() * dims_b.W() * batch_size, dtype=np.float32)
    output_c = cuda.pagelocked_empty(dims_c.C() * dims_c.H() * dims_c.W() * batch_size, dtype=np.float32)


    # alocate device memory
    a_input = cuda.mem_alloc(batch_size * dims_a.C() * dims_a.H() * dims_a.W() * input_data.dtype.itemsize)
    b_output = cuda.mem_alloc(batch_size * dims_b.C() * dims_b.H() * dims_b.W() * output_b.dtype.itemsize)
    c_output = cuda.mem_alloc(batch_size * dims_c.C() * dims_c.H() * dims_c.W() * output_c.dtype.itemsize)

    bindings = [int(a_input), int(b_output), int(c_output)]

    stream = cuda.Stream()

    # transfer input data to device
    cuda.memcpy_htod_async(a_input, input_data, stream)  # feed
    # execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # transfer predictions back
    cuda.memcpy_dtoh_async(output_b, b_output, stream)  # fetch
    cuda.memcpy_dtoh_async(output_c, c_output, stream)  # fetch

    # return predictions
    print "output_b = ", output_b
    print "output_c = ", output_c
    return output_b, output_c


if __name__ == "__main__":
    tf_model = get_frozen_graph()
    # test_frozen_graph(tf_model)
    tensorrt_infer(tf_model)


