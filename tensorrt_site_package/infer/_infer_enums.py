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

'''
Enum definitions from libnvinfer
'''

from tensorrt.infer import _nv_infer_bindings as nvinfer
from enum import IntEnum
import numpy as np

class DataType(IntEnum):
    '''Available data types

    Base Class:
        IntEnum
    '''
    FLOAT = nvinfer.DataType_kFLOAT
    HALF  = nvinfer.DataType_kHALF
    INT8  = nvinfer.DataType_kINT8
    INT32 = nvinfer.DataType_kINT32

    def size(self):
        '''Returns:
            - ``int``: Size in bytes
        '''
        if self.value == 0:    #FP32
            return 4
        elif self.value == 1:  #FP16
            return 2
        elif self.value == 2:  #INT8
            return 1
        elif self.value == 3:  #INT32
            return 4
        else:
            raise ValueError("unknown DataType")

    @property
    def input_type(self):
        '''Returns:
            - ``numpy type``: expected data type for input
        '''
        if self.value == 0:    #FP32
            return np.float32
        elif self.value == 1:  #FP16
            return np.float16
        elif self.value == 2:  #INT8
            return np.float32
        elif self.value == 3:  #INT32
            return np.float32
        else:
            raise ValueError("unknown DataType")

    @property
    def nptype(self):
        '''Returns:
            - ``numpy type``: Analogous numpy type
        '''
        if self.value == 0:    #FP32
            return np.float32
        elif self.value == 1:  #FP16
            return np.float16
        elif self.value == 2:  #INT8
            return np.int8
        elif self.value == 3:  #INT32
            return np.int32
        else:
            raise ValueError("unknown DataType")


class DimensionType(IntEnum):
    '''Available dimension types

    Base Class:
        IntEnum
    '''
    SPATIAL  = nvinfer.DimensionType_kSPATIAL
    CHANNEL  = nvinfer.DimensionType_kCHANNEL
    INDEX    = nvinfer.DimensionType_kINDEX
    SEQUENCE = nvinfer.DimensionType_kSEQUENCE

class LayerType(IntEnum):
    '''Available layer types

    Base Class:
        IntEnum
    '''
    CONVOLUTION     = nvinfer.LayerType_kCONVOLUTION
    FULLY_CONNECTED = nvinfer.LayerType_kFULLY_CONNECTED
    ACTIVATION      = nvinfer.LayerType_kACTIVATION
    POOLING         = nvinfer.LayerType_kPOOLING
    LRN             = nvinfer.LayerType_kLRN
    SCALE           = nvinfer.LayerType_kSCALE
    SOFTMAX         = nvinfer.LayerType_kSOFTMAX
    DECONVOLUTION   = nvinfer.LayerType_kDECONVOLUTION
    CONCATENATION   = nvinfer.LayerType_kCONCATENATION
    ELEMENTWISE     = nvinfer.LayerType_kELEMENTWISE
    PLUGIN          = nvinfer.LayerType_kPLUGIN
    RNN             = nvinfer.LayerType_kRNN
    UNARY           = nvinfer.LayerType_kUNARY
    PADDING         = nvinfer.LayerType_kPADDING
    SHUFFLE         = nvinfer.LayerType_kSHUFFLE

class ActivationType(IntEnum):
    '''Type of activation function

    Base Class:
        IntEnum
    '''
    RELU    = nvinfer.ActivationType_kRELU
    SIGMOID = nvinfer.ActivationType_kSIGMOID
    TANH    = nvinfer.ActivationType_kTANH

class PoolingType(IntEnum):
    '''Type of pooling layer

    Base Class:
        IntEnum
    '''
    MAX     = nvinfer.PoolingType_kMAX
    AVERAGE = nvinfer.PoolingType_kAVERAGE
    MAX_AVERAGE_BLEND = nvinfer.PoolingType_kMAX_AVERAGE_BLEND

class ScaleMode(IntEnum):
    '''Scale mode

    Base Class:
        IntEnum
    '''
    UNIFORM     = nvinfer.ScaleMode_kUNIFORM
    CHANNEL     = nvinfer.ScaleMode_kCHANNEL
    ELEMENTWISE = nvinfer.ScaleMode_kELEMENTWISE

class ElementWiseOperation(IntEnum):
    '''Type of operation for the layer

    Base Class:
        IntEnum
    '''
    SUM  = nvinfer.ElementWiseOperation_kSUM
    PROD = nvinfer.ElementWiseOperation_kPROD
    MAX  = nvinfer.ElementWiseOperation_kMAX
    MIN  = nvinfer.ElementWiseOperation_kMIN
    SUB  = nvinfer.ElementWiseOperation_kSUB
    DIV  = nvinfer.ElementWiseOperation_kDIV
    POW  = nvinfer.ElementWiseOperation_kPOW

class RNNOperation(IntEnum):
    '''Type of operation for the layer

    Base Class:
        IntEnum
    '''
    RELU = nvinfer.RNNOperation_kRELU
    TANH = nvinfer.RNNOperation_kTANH
    LSTM = nvinfer.RNNOperation_kLSTM
    GRU  = nvinfer.RNNOperation_kGRU

class RNNDirection(IntEnum):
    '''Direction for the RNN Layer

    Base Class:
        IntEnum
    '''
    UNIDIRECTION = nvinfer.RNNDirection_kUNIDIRECTION
    BIDIRECTION  = nvinfer.RNNDirection_kBIDIRECTION

class RNNInputMode(IntEnum):
    '''Input mode for RNN Layer

    Base Class:
        IntEnum
    '''
    LINEAR = nvinfer.RNNInputMode_kLINEAR
    SKIP   = nvinfer.RNNInputMode_kSKIP


class UnaryOperation(IntEnum):
    '''Type of operation for the layer

    Base Class:
        IntEnum
    '''
    EXP = nvinfer.UnaryOperation_kEXP
    LOG = nvinfer.UnaryOperation_kLOG
    SQRT = nvinfer.UnaryOperation_kSQRT
    RECIP = nvinfer.UnaryOperation_kRECIP
    ABS = nvinfer.UnaryOperation_kABS
    NEG = nvinfer.UnaryOperation_kNEG

class CalibrationAlgoType(IntEnum):
    '''Type of int8 calibration algorithm

    Base Class:
        IntEnum
    '''
    LEGACY_CALIBRATION  = nvinfer.CalibrationAlgoType_kLEGACY_CALIBRATION
    ENTROPY_CALIBRATION = nvinfer.CalibrationAlgoType_kENTROPY_CALIBRATION

#TODO:Move this into Logger somehow
class LogSeverity(IntEnum):
    '''Log level specifier

    Base Class:
        IntEnum
    '''
    INTERNAL_ERROR = nvinfer.Logger.Severity_kINTERNAL_ERROR
    ERROR          = nvinfer.Logger.Severity_kERROR
    WARNING        = nvinfer.Logger.Severity_kWARNING
    INFO           = nvinfer.Logger.Severity_kINFO
