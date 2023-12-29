# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Intrinsics of TVM-Python Hybrid Script for Python emulation runtime"""

# 2022.02.15 - Support grid as new loop mode.
# 2021.12.15 - Support block_realize intrin; update the def of WithStub in runtime.
# 2021.12.02 - Support more math intrin.
# 2019.12.30 - Add class WithStub and support more functions in HYBRID_GLOBALS.

from itertools import product
import numpy
from .. import target

class bind(object): #pylint: disable=invalid-name
    """GPU bind software emulataion runtime."""
    def __init__(self, _, ext):
        self.ext = ext

    def __iter__(self):
        i = 0
        while i < self.ext:
            yield i
            i += 1


def allocate(shape, dtype='float32', scope='global'): #pylint: disable=unused-argument
    """Allocate a buffer with given shape

    Parameters
    ----------
    shape: Tuple
        The shape of the tensor to be allocated
    dtype: string
        The data type of the tensor
    scope: string
        The storage scope of the tensor

    Returns
    -------
    tensor: numpy.array
        The tensor allocated
    """
    return numpy.zeros(shape).astype(dtype)


def rsqrt(x):
    """
    Computes reciprocal of square root of x element-wise

    Parameters
    ----------
    x: Tensor

    Returns
    -------
    res: Tensor
        The result of reciprocal of square root of x
    """
    return numpy.ones_like(x) / numpy.sqrt(x)


def popcount(x):
    """
    Count ones in the binary representation of number x

    Parameters
    ----------
    x: Integer
        The number to be counted

    Returns
    -------
    cnt: Integer
        The number of ones in the binary representation of number x
    """
    cnt = 0
    while x:
        x -= x & -x
        cnt += 1
    return cnt


def sigmoid(x):
    """
    Sigmoid function of x, aka 1/(1+exp(-x)).

    Parameters
    ----------
    x: a real number

    Returns
    -------
    res: a real number
        The result of sigmoid function
    """
    return 1 / (1 + numpy.exp(-x))

def erf(x):
    """
    Erf function of x, aka erf(x) = 2 / sqrt(pi) * integral(exp(-t*t), t = 0..x).
    The algorithm comes from Handbook of Mathematical Functions, formula 7.1.26.

    Parameters
    ----------
    x: a real number

    Returns
    -------
    res: a real number
        The result of sigmoid function
    """
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = numpy.abs(x)

    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911

    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*numpy.exp(-x*x)
    return sign*y # erf(-x) = -erf(x)

def max_num_threads(allow_none=True):
    """Get max number of threads for GPU targets."""
    return target.current_target(allow_none).max_num_threads


def grid(extents):
    extents_list = []
    for ext in extents:
        extents_list.append(range(ext))
    return product(*extents_list)


class WithStub:

    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        return self

    def __del__(self):
        return self

    def __call__(self, *arg, **kwargs):
        return self

HYBRID_GLOBALS = {
    'unroll'         : range,
    'vectorize'      : range,
    'parallel'       : range,
    'const_range'    : range,
    'serial'         : range,
    'reduce'         : range,
    'bind'           : bind,
    'allocate'       : allocate,
    'output_tensor'  : allocate,
    'grid'           : grid,
    'sqrt'           : numpy.sqrt,
    'rsqrt'          : rsqrt,
    'sign'           : numpy.sign,
    'sin'            : numpy.sin,
    'cos'            : numpy.cos,
    'isnan'          : numpy.isnan,
    'isinf'          : numpy.isinf,
    'isfinite'       : numpy.isfinite,
    'erf'            : erf,
    'atan'           : numpy.arctan,
    'atan2'          : numpy.arctan2,
    'log'            : numpy.log,
    'tanh'           : numpy.tanh,
    'power'          : numpy.power,
    'exp'            : numpy.exp,
    'expm1'          : numpy.expm1,
    'sigmoid'        : sigmoid,
    'popcount'       : popcount,
    'floor'          : numpy.floor,
    'ceil'           : numpy.ceil,
    'trunc'          : numpy.trunc,
    'abs'            : numpy.abs,
    'round'          : numpy.round,
    'likely'         : lambda cond: cond,
    'uint8'          : numpy.uint8,
    'uint16'         : numpy.uint16,
    'uint32'         : numpy.uint32,
    'uint64'         : numpy.uint64,
    'int8'           : numpy.int8,
    'int16'          : numpy.int16,
    'int32'          : numpy.int32,
    'int64'          : numpy.int64,
    'float16'        : numpy.float16,
    'float32'        : numpy.float32,
    'float64'        : numpy.float64,
    'ceil_div'       : lambda a, b: (a + b - 1) // b,
    'attr'           : WithStub(),
    'block_realize'  : WithStub(),
    'max_num_threads': max_num_threads,
    'sub_relu': lambda a, b: a
}


def _enter_hybrid_runtime(func):
    """Put hybrid runtime variables into the global scope"""
    _globals = func.__globals__
    intersect = []
    for elem in list(HYBRID_GLOBALS.keys()):
        if elem in _globals.keys():
            intersect.append((elem, _globals[elem]))
        _globals[elem] = HYBRID_GLOBALS[elem]
    return intersect


def _restore_runtime(func, intersect):
    """Rollback the modification caused by hybrid runtime"""
    _globals = func.__globals__
    for elem in list(HYBRID_GLOBALS.keys()):
        _globals.pop(elem)
    for k, v in intersect:
        _globals[k] = v
