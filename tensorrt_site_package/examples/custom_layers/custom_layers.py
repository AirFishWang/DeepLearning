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


from random import randint
import numpy as np

try:
    from PIL import Image
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have Pillow installed.
For installation instructions, see:
http://pillow.readthedocs.io/en/stable/installation.html""".format(err))

try:
    import subprocess
    # Make sure we have swig 3.X
    ver = subprocess.check_output(["swig", "-version"]).decode().find("Version 3")
    assert(ver != -1)
except AssertionError as err:
    # AssertionError occurs when swig is installed but version != 3.
    raise AssertionError("""ERROR: An incompatible version of Swig is installed.
Please install Swig 3.0 from here:
http://www.swig.org/Doc3.0/Preface.html#Preface_installation""")
except OSError as err:
    # OSError when swig is not installed.
    raise OSError("""ERROR: Swig is not installed.
Please see installation instructions here:
http://www.swig.org/Doc3.0/Preface.html#Preface_installation""")

try:
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import argparse
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]""".format(err))

try:
    import tensorrt as trt
    from tensorrt.parsers import caffeparser
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH.""".format(err))

try:
    import tensorrtplugins
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please build and install the example custom layer wrapper
Follow the instructions in the README on how to do so""".format(err))

G_LOGGER = trt.infer.ColorLogger(trt.infer.LogSeverity.INFO)
INPUT_LAYERS = ["data"]
OUTPUT_LAYERS = ['prob']
INPUT_H = 28
INPUT_W =  28
OUTPUT_SIZE = 10

PARSER = argparse.ArgumentParser(description="Example of how to create a Caffe based TensorRT Engine, use a custom layer implementation and run inference")
PARSER.add_argument('datadir', help='Path to Python TensorRT data directory (realpath)')

ARGS = PARSER.parse_args()
DATA_DIR = ARGS.datadir

MODEL_PROTOTXT = DATA_DIR + '/mnist/mnist.prototxt'
CAFFE_MODEL =  DATA_DIR + '/mnist/mnist.caffemodel'
DATA =  DATA_DIR + '/mnist/'
IMAGE_MEAN =  DATA_DIR + '/mnist/mnist_mean.binaryproto'

# Run inference on device
def infer(context, input_img, output_size, batch_size):
    # Load engine
    engine = context.get_engine()
    assert(engine.get_nb_bindings() == 2)
    # Convert input data to Float32
    input_img = input_img.astype(np.float32)
    # Create output array to receive data
    output = np.empty(output_size, dtype = np.float32)

    # Alocate device memory
    d_input = cuda.mem_alloc(batch_size * input_img.size * input_img.dtype.itemsize)
    d_output = cuda.mem_alloc(batch_size * output.size * output.dtype.itemsize)

    bindings = [int(d_input), int(d_output)]

    stream = cuda.Stream()

    # Transfer input data to device
    cuda.memcpy_htod_async(d_input, input_img, stream)
    # Execute model
    context.enqueue(batch_size, bindings, stream.handle, None)
    # Transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    # Return predictions
    return output

def get_testcase(path):
    im = Image.open(path)
    assert(im)
    # Make array 1D
    return np.array(im).ravel()

def apply_mean(img, mean_path):
    parser = caffeparser.create_caffe_parser()
    # Parse mean
    mean_blob = parser.parse_binary_proto(mean_path)
    parser.destroy()
    # Note: In TensorRT C++ no size is reqired, however you need it to cast the array
    mean = mean_blob.get_data(INPUT_W * INPUT_H)

    data = np.empty([INPUT_H * INPUT_W])
    for i in range(INPUT_W * INPUT_H):
        data[i] = float(img[i]) - mean[i]

    mean_blob.destroy()
    return data

def main():
    plugin_factory = tensorrtplugins.FullyConnectedPluginFactory()

    # Convert caffe model to TensorRT engine
    runtime = trt.infer.create_infer_runtime(G_LOGGER)
    engine = trt.utils.caffe_to_trt_engine(G_LOGGER,
        MODEL_PROTOTXT,
        CAFFE_MODEL,
        1,
        1 << 20,
        OUTPUT_LAYERS,
        trt.infer.DataType.FLOAT,
        plugin_factory)

    # Get random test case
    rand_file = randint(0, 9)
    img = get_testcase(DATA + str(rand_file) + '.pgm')

    print("Test case: " + str(rand_file))
    data = apply_mean(img, IMAGE_MEAN)

    context = engine.create_execution_context()

    out = infer(context, data, OUTPUT_SIZE, 1)

    print("Prediction: " + str(np.argmax(out)))

    # Destroy engine
    context.destroy()
    engine.destroy()
    runtime.destroy()

    # Destroy plugin created by factory
    plugin_factory.destroyPlugin()

if __name__ == "__main__":
    main()
