# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:    convert_pb
   Description:
   Author:       wangchun
   date:         2020/8/25
-------------------------------------------------
"""
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from tensorflow.python.platform import gfile
from tensorflow.python.framework import graph_io
from tensorflow.python.framework.graph_util import convert_variables_to_constants


def convert_keras_to_pb():
    K.set_learning_phase(0)
    # model_path = "model_cnn.h5"
    # model_path = "model_lstm.h5"
    model_path = "path of keras h5 model"


    model = load_model(model_path)

    print('input is:', [i.op.name for i in model.inputs])
    print('output is:', [i.op.name for i in model.outputs])

    output_node = model.outputs
    sess = K.get_session()
    graph = sess.graph
    with graph.as_default():
        input_graph_def = graph.as_graph_def()
        frozen_graph = convert_variables_to_constants(sess, input_graph_def, [i.op.name for i in output_node])
        frozen_graph = tf.graph_util.remove_training_nodes(frozen_graph)

    graph_io.write_graph(frozen_graph, './', model_path.replace(".h5", ".pb"), as_text=False)
    print("convert .h5 to .pb finished")
    exit()


model_path = "path of pb model"

with gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

cnn_input = 'input_2:0'
cnn_output = 'global_average_pooling2d_1/Mean:0'
graph = tf.Graph()
with graph.as_default():
    cnn_input, cnn_output = tf.import_graph_def(
        graph_def=graph_def, name="",
        return_elements=[cnn_input, cnn_output])
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess_cnn = tf.Session(config=config)