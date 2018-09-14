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
TensorRT Python API
'''
from tensorrt import __versions__
__version__ = __versions__.package_version
__backend_version__ = __versions__.infer_lib_version
__backend_so_version__ = __versions__.so_version
__cudnn_version__ = __versions__.cudnn_version

import sys
import os as _dl_flags

if not hasattr(_dl_flags, 'RTLD_GLOBAL') or not hasattr(_dl_flags, 'RTLD_NOW'):
    try:
        import DLFCN as _dl_flags
    except ImportError:
        pass

old_flags = sys.getdlopenflags()
sys.setdlopenflags(_dl_flags.RTLD_GLOBAL | _dl_flags.RTLD_NOW)

from tensorrt import infer, parsers, utils, lite, plugins

inferLibMajor = int(str(infer.get_infer_lib_version())[:1])
inferLibMinor = int(str(infer.get_infer_lib_version())[1:2])
inferLibPatch = int(str(infer.get_infer_lib_version())[2:])

requiredBackendVersionMajor = int(str(__backend_version__)[:1])
requiredBackendVersionMinor = int(str(__backend_version__)[1:2])
requiredBackendVersionPatch = int(str(__backend_version__)[2:])

if inferLibMajor != requiredBackendVersionMajor or inferLibMinor != requiredBackendVersionMinor or inferLibPatch != requiredBackendVersionPatch:
    raise ImportError('TensorRT Library mismatch, expected version ' + str(__version__) + ' got version ' + str(inferLibMajor) + '.' + str(inferLibMinor) + '.' + str(inferLibPatch))

'''
if __backend_version__ != int(str(infer.get_infer_lib_version())):
    print('__backend_version__ = %d,   infer.get_infer_lib_version = %d'  % (__backend_version__, int(str(infer.get_infer_lib_version()))))
    raise ImportError('TensorRT Library mismatch, verify that the correct library version is installed, expecting libnvinfer.so.' + str(__backend_so_version__))
'''

sys.setdlopenflags(old_flags)
del _dl_flags
del old_flags
