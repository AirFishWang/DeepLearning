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

from __future__ import print_function
import os
import sys
import traceback
import tensorrt
from tensorrt import infer
from tensorrt.parsers import caffeparser
from tensorrt.parsers import uffparser
from tensorrt.utils import _nv_utils_bindings as nvutils


def trt_network_to_trt_engine(logger, network, max_batch_size, max_workspace_size, datatype=infer.DataType.FLOAT, plugin_factory=None, calibrator=None):
    '''
    Takes in TensorRT network as an input, set the rest of parameters for the network and generates an engine

    Does not destroy Network

    Args:
        - **logger** ``tensorrt.infer.Logger``: Logging system for the application
        - **trt_graph** ``str``: network
        - **max_batch_size** ``int``: Maximum batch size
        - **max_workspace_size** ``int``: Maximum workspace size
        - **datatype** ``tensorrt.infer.DataType``: Operating data type of the engine, can be FP32, FP16 if supported on the platform, or INT8 with calibrator. Default: ``tensorrt.infer.DataType.FLOAT``
        - **plugins_factory** ``tensorrt.infer.PluginFactory`` *(optional)*: Custom layer factory. Default: `` None``
        - **calibrator** ``tensorrt.infer.Int8Calibrator`` *(optional)*: Currently unsupported. Default: ``None``

    Returns:
        - ``tensorrt.infer.CudaEngine``: TensorRT Engine to be used or excuted
    '''
    builder = infer.create_infer_builder(logger)


    if datatype == infer.DataType.HALF and not builder.platform_has_fast_fp16():
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Specified FP16 but not supported on platform")
        raise AttributeError("Specified FP16 but not supported on platform")

    if datatype == infer.DataType.INT8 and calibrator == None:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Specified INT8 but no calibrator provided")
        raise AttributeError("Specified INT8 but no calibrator provided")


    if datatype == infer.DataType.INT32:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "INT32 is not supported at this time ")
        raise AttributeError("INT32 is not supported at this time")



    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(max_workspace_size)

    if datatype == infer.DataType_kHALF:
        builder.set_fp16_mode(True)

    if datatype == infer.DataType_kINT8 and calibrator:
        builder.set_debug_sync(True)
        builder.set_int8_mode(True)
        builder.set_int8_calibrator(calibrator)

    engine = builder.build_cuda_engine(network)

    try:
        assert(engine)
    except AssertionError:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Failed to create engine")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        raise AssertionError('Engine build failed on line {} in statement {}'.format(line, text))

    builder.destroy()
    return engine


def uff_file_to_trt_engine(logger, uff_file, parser, max_batch_size, max_workspace_size, datatype=infer.DataType.FLOAT, plugin_factory=None, calibrator=None):
    '''
    Parses a UFF file and generates an engine

    Takes a UFF file (created with a UFF exporter) and generates a TensorRT engine that
    can then be saved or executed

    Args:
        - **logger** ``tensorrt.infer.Logger``: Logging system for the application
        - **uff_file** ``str``: Path to UFF file
        - **parser** ``tensorrt.parsers.uffparser.UffParser``: uff parser
        - **max_batch_size** ``int``: Maximum batch size
        - **max_workspace_size** ``int``: Maximum workspace size
        - **datatype** ``tensorrt.infer.DataType``: Operating data type of the engine, can be FP32, FP16 if supported on the platform, or INT8 with calibrator. Default: ``tensorrt.infer.DataType.FLOAT``
        - **plugins_factory** ``tensorrt.infer.PluginFactory`` *(optional)*: Custom layer factory. Default: `` None``
        - **calibrator** ``tensorrt.infer.Int8Calibrator`` *(optional)*: Currently unsupported. Default: ``None``

    Returns:
        - ``tensorrt.infer.CudaEngine``: TensorRT Engine to be used or excuted
    '''
    builder = infer.create_infer_builder(logger)
    network = builder.create_network()

    if plugin_factory:
        parser.set_plugin_factory(plugin_factory)

    if datatype == infer.DataType.HALF and not builder.platform_has_fast_fp16():
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Specified FP16 but not supported on platform")
        raise AttributeError("Specified FP16 but not supported on platform")
        return

    if datatype == infer.DataType.INT8 and calibrator == None:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Specified INT8 but no calibrator provided")
        raise AttributeError("Specified INT8 but no calibrator provided")

    if datatype == infer.DataType.INT32:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "INT32 is not supported in uff parser at this time ")
        raise AttributeError("INT32 is not supported in uff parser at this time")

    model_datatype = infer.DataType_kFLOAT
    if datatype == infer.DataType_kHALF:
        model_datatype = infer.DataType_kHALF

    try:
        assert(parser.parse_from_file(uff_file, network, model_datatype))
    except AssertionError:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Failed to parse UFF File '{}'".format(uff_file))
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        raise AssertionError('UFF parsing failed on line {} in statement {}'.format(line, text))

    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(max_workspace_size)

    if datatype == infer.DataType_kHALF:
        builder.set_fp16_mode(True)

    if datatype == infer.DataType_kINT8:
        builder.set_average_find_iterations(1)
        builder.set_min_find_iterations(1)
        builder.set_debug_sync(True)
        builder.set_int8_mode(True)
        builder.set_int8_calibrator(calibrator)

    engine = builder.build_cuda_engine(network)

    try:
        assert(engine)
    except AssertionError:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Failed to create engine")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        raise AssertionError('UFF parsing failed on line {} in statement {}'.format(line, text))

    network.destroy()
    builder.destroy()

    return engine

def uff_to_trt_engine(logger, stream, parser, max_batch_size, max_workspace_size, datatype=infer.DataType.FLOAT, plugin_factory=None, calibrator=None):
    '''
    Parses a UFF Model Stream and generates an engine

    Takes a UFF Stream (created with a UFF exporter) and generates a TensorRT engine that
    can then be saved or executed

    Args:
        - **logger** ``tensorrt.infer.Logger``: Logging system for the application
        - **stream** ``[Py2]str/[Py3]bytes``: Serialized UFF graph
        - **parser** ``tensorrt.parsers.uffparser.UffParser``: uff parser
        - **max_batch_size** ``int``: Maximum batch size
        - **max_workspace_size** ``int``: Maximum workspace size
        - **datatype** ``tensorrt.infer.DataType``: Operating data type of the engine, can be FP32, FP16 if supported on the platform, or INT8 with calibrator. Default: ``tensorrt.infer.DataType.FLOAT``
        - **plugins_factory** ``tensorrt.infer.PluginFactory`` *(optional)*: Custom layer factory. Default: ``None``
        - **calibrator** ``tensorrt.infer.Int8Calibrator`` *(optional)*: Currently unsupported. Default: ``None``

    Returns:
        - ``tensorrt.infer.CudaEngine``: TensorRT Engine to be used or excuted

    '''
    builder = infer.create_infer_builder(logger)
    network = builder.create_network()

    if plugin_factory:
        parser.set_plugin_factory(plugin_factory)

    if datatype == infer.DataType.HALF and not builder.platform_has_fast_fp16():
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Specified FP16 but not supported on platform")
        raise AttributeError("Specified FP16 but not supported on platform")
        return

    if datatype == infer.DataType.INT8 and calibrator == None:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Specified INT8 but no calibrator provided")
        raise AttributeError("Specified INT8 but no calibrator provided")

    if datatype == infer.DataType.INT32:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "INT32 is not supported in uff parser at this time ")
        raise AttributeError("INT32 is not supported in uff parser at this time")

    model_datatype = infer.DataType_kFLOAT
    if datatype == infer.DataType_kHALF:
        model_datatype = infer.DataType_kHALF

    try:
        assert(parser.parse(stream, network, model_datatype))
    except AssertionError:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Failed to parse UFF model stream")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        raise AssertionError('UFF parsing failed on line {} in statement {}'.format(line, text))



    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(max_workspace_size)

    if datatype == infer.DataType_kHALF:
        builder.set_fp16_mode(True)

    if datatype == infer.DataType_kINT8:
        builder.set_average_find_iterations(1)
        builder.set_min_find_iterations(1)
        builder.set_debug_sync(True)
        builder.set_int8_mode(True)
        builder.set_int8_calibrator(calibrator)

    engine = builder.build_cuda_engine(network)

    try:
        assert(engine)
    except AssertionError:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Failed to create engine")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        raise AssertionError('UFF parsing failed on line {} in statement {}'.format(line, text))


    network.destroy()
    builder.destroy()

    return engine

#Parse Caffe files and generate an engine
def caffe_to_trt_engine(logger, deploy_file, model_file, max_batch_size, max_workspace_size, output_layers, datatype=infer.DataType.FLOAT, plugin_factory=None, calibrator=None):
    """Parses a Caffe model and create an engine for inference

    Takes a Caffe model prototxt and caffemodel, name(s) of the output layer(s), and engine settings
    to create a engine that can be used for inference

    Args:
        - **logger** ``tensorrt.infer.Logger``: A logger is needed to monitor the progress of building the engine
        - **deploy_file** ``str``: Path to caffe model prototxt
        - **model_file** ``str``: Path to caffe caffemodel file
        - **max_batch_size** ``int``: Maximum size of batch allowed for the engine
        - **max_workspace_size** ``int``: Maximum size of engine maxWorkspaceSize
        - **output_layers** ``[str]``: List of output layer names
        - **datatype** ``tensorrt.infer.DataType``: Operating data type of the engine, can be FP32, FP16 if supported on the platform, or INT8 with calibrator. Default: ``tensorrt.infer.DataType.FLOAT``
        - **plugin_factory** ``tensort.infer.PluginFactory``  *(optional)*: Custom layer factory. Default:``None``
        - **calibrator** ``INT8 calibrator`` *(optional)*: (currently unsupported in python). Default:``None``

    Returns
        - ``tensorrt.infer.CudaEngine``: An engine that can be used to execute inference

    """


    #create the builder
    builder = infer.create_infer_builder(logger)

    #parse the caffe model to populate the network
    network = builder.create_network()
    parser = caffeparser.create_caffe_parser()

    if plugin_factory:
        parser.set_plugin_factory(plugin_factory)

    #TODO: Generalize for different data types
    if datatype == infer.DataType_kHALF and not builder.platform_has_fast_fp16():
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Specified FP16 but not supported on platform")
        raise AttributeError("Specified FP16 but not supported on platform")
        return

    if datatype == infer.DataType_kINT8 and calibrator == None:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Specified INT8 but no calibrator provided")
        raise AttributeError("Specified INT8 but no calibrator provided")
        return

    model_datatype = infer.DataType_kFLOAT
    if datatype == infer.DataType_kHALF:
        model_datatype = infer.DataType_kHALF

    blob_name_to_tensor = parser.parse(deploy_file, model_file, network, model_datatype)
    logger.log(tensorrt.infer.LogSeverity.INFO, "Parsing caffe model {}, {}".format(deploy_file, model_file))

    try:
        assert(blob_name_to_tensor)
    except AssertionError:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Failed to parse caffe model")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        raise AssertionError('Caffe parsing failed on line {} in statement {}'.format(line, text))

    input_dimensions = {}

    for i in range(network.get_nb_inputs()):
        dims = network.get_input(i).get_dimensions().to_DimsCHW()
        logger.log(tensorrt.infer.LogSeverity.INFO, "Input \"{}\":{}x{}x{}".format(network.get_input(i).get_name(), dims.C(), dims.H(), dims.W()))
        input_dimensions[network.get_input(i).get_name()] = network.get_input(i).get_dimensions().to_DimsCHW()

    if type(output_layers) is str:
        output_layers = [output_layers]

    #mark the outputs
    for l in output_layers:
        logger.log(tensorrt.infer.LogSeverity.INFO, "Marking " + l + " as output layer")
        t = blob_name_to_tensor.find(l)
        try:
            assert(t)
        except AssertionError:
            logger.log(tensorrt.infer.LogSeverity.ERROR, "Failed to find output layer {}".format(l))
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb) # Fixed format
            tb_info = traceback.extract_tb(tb)
            filename, line, func, text = tb_info[-1]

            raise AssertionError('Caffe parsing failed on line {} in statement {}'.format(line, text))

        layer = network.mark_output(t)

    for i in range(network.get_nb_outputs()):
        dims = network.get_output(i).get_dimensions().to_DimsCHW()
        logger.log(tensorrt.infer.LogSeverity.INFO, "Output \"{}\":{}x{}x{}".format(network.get_output(i).get_name(), dims.C(), dims.H(), dims.W()))

    #build the engine
    builder.set_max_batch_size(max_batch_size)
    builder.set_max_workspace_size(max_workspace_size)

    if datatype == infer.DataType_kHALF:
        logger.log(tensorrt.infer.LogSeverity.INFO, "Enabling FP16 mode")
        builder.set_fp16_mode(True)

    if datatype == infer.DataType_kINT8:
        builder.set_average_find_iterations(1);
        builder.set_min_find_iterations(1);
        builder.set_debug_sync(True);
        builder.set_int8_mode(True);
        builder.set_int8_calibrator(calibrator);

    logger.log(tensorrt.infer.LogSeverity.INFO, "Building engine")

    engine = builder.build_cuda_engine(network)

    try:
        assert(engine)
    except AssertionError:
        logger.log(tensorrt.infer.LogSeverity.ERROR, "Failed to create engine")
        _, _, tb = sys.exc_info()
        traceback.print_tb(tb) # Fixed format
        tb_info = traceback.extract_tb(tb)
        filename, line, func, text = tb_info[-1]

        raise AssertionError('Caffe parsing failed on line {} in statement {}'.format(line, text))

    network.destroy()
    parser.destroy()
    builder.destroy()

    return engine

#Expose loadEngine C++ in python
def load_engine(logger, filepath, plugins = None):
    """Load a saved engine file

    Creates an engine from a file containting a serialized engine

    Args:
        - **logger** ``tensorrt.infer.Logger``: A logger is needed to monitor the progress of building the engine
        - **filepath** ``str``: Path to engine file
        - **plugins** ``tensorrt.infer.PluginFactory`` *(optional)*: Custom layer factory

    Returns:
        - ``tensorrt.infer.CudaEngine``: An engine that can be used to execute inference

    """
    filepath = _file_exists(filepath)
    return nvutils.cload_engine(logger, filepath, plugins)

#Expose writeEngine C++ in python
def write_engine_to_file(filepath, engine):
    """Write an engine to a file

    Takes a serialized engine and wrties it to a file to be loaded
    later

    Args:
        - **filepath** ``str``: Path to engine file
        - **engine** ``tensorrt.infer.CudaEngine``: An engine that can be used to execute inference

    Returns:
        - ``bool``: Whether the file was written or not

    """
    return nvutils.cwrite_engine_to_file(filepath, engine)

#Expose loadWeights from wts file
def load_weights(filepath):
    """Load model weights from file

    Loads weights from a .wts file into a dictionary of layer names
    and associated weights encoded in tensorrt.infer.Weights object

    Args:
        - **filepath** ``str``: path to the weights file

    Returns:
        - ``dict {str, tensorrt.infer.Weights}``: Dictionary of layer names and associated weights

    """
    filepath = _file_exists(filepath)
    return nvutils.cload_weights(filepath)

def _file_exists(path):
    path = str(path)
    if not os.path.isfile(path):
        raise ValueError("File does not exist")
    return path

#TODO Expose rest of NvUtils Functions here featuring numpy arrays

