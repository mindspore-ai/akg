# Copyright 2019-2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Data dump load operation
"""
import os
import struct
import numpy as np
from tests.common.log import Log


def dump_tensor(tensor, file_path):
    if isinstance(tensor, np.ndarray):
        np.save(file_path, tensor)
    else:
        rank = len(tensor.shape)
        t = (rank,) + tensor.shape
        desc = struct.pack('I%dI' % rank, *t)
        with open(file_path, 'wb') as f:
            f.write(desc)
            f.write(tensor.tobytes())


def load_tensor(file_path, dtype=None):
    if file_path.endswith(".npy"):
        data = np.load(file_path)
        return len(data.shape), data.shape, data
    else:
        with open(file_path, 'rb') as f:
            content = f.read()
        rank = struct.unpack_from('I', content, 0)[0]
        shape = struct.unpack_from('%dI' % rank, content, 4)
        data = np.frombuffer(
            content[4 + rank * 4:], dtype=dtype).reshape(shape)
    return rank, shape, data


def compare_tensor(acu_output, exp_output, rtol=1.e-5, atol=1.e-8, equal_nan=False):
    """
    Output and expected result comparison method
    :param acu_output: array_like Input arrays to compare.
    :param exp_output: array_like Input arrays to compare.
    :param rtol: float The relative tolerance parameter (see Notes).
    :param atol: float The absolute tolerance parameter (see Notes).
    :param equal_nan: bool
            Whether to compare NaN's as equal.  If True, NaN's in `a` will be
            considered equal to NaN's in `b` in the output array.
    :return: True / False
    """
    res = np.allclose(acu_output, exp_output, rtol, atol, equal_nan)
    if not res:
        pandora_logger_ = Log(case_name=os.path.dirname(__file__), case_path=os.getcwd())
        pandora_logger_.log.error("This shape precision is not up to standard, compare failed.")
    return res
