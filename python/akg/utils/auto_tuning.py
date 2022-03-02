#!/usr/bin/env python3
# coding: utf-8
# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
tuning utils
"""
from typing import Tuple
import struct
import numpy as np

import akg.tvm

# The length of the feature vector
DEFAULT_FEATURE_VEC_LEN = 164

# The size of int and float in bytes
SIZE_OF_INT32 = 4
SIZE_OF_FLOAT32 = 4


def unpack_feature(byte_arr: bytearray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack the flatten feature (in byte array format) from c++

    Parameters
    ----------
    byte_arr: bytearray
        The two-dimensional feature vector in serialized byte array format

    Returns
    -------
    features: np.ndarray
        Feature vectors
    normalized_throughputs: np.ndarray
        Normalized throughputs
    task_ids: np.ndarray
        Task ids

    Note
    ----
    For faster data copy between c++ and python, the c++ part returns features in a single
    flatten array using a packed format. The python part then unpacks the flatten array.

    The packed format for n records is:
    {
      int   n;
      int   sizes[n+1];           // The sizes for the following arrays

      float features_0[size[0]];  // The features for record 0
      float features_1[size[1]];  // The features for record 1
      ...
      float features_i[size[i]];  // The features for record i
      ... // until i == n - 1

      float throughputs[sizes[n]];  // The normalized throughputs for n records

    }
    To implement this format, we also store int as float, so we can store all numbers
    into a single float array.
    """
    vec_len = DEFAULT_FEATURE_VEC_LEN

    # unpack sizes
    offset = 0
    n = struct.unpack_from("1i", byte_arr, offset=offset)[0]
    offset += SIZE_OF_INT32

    sizes = struct.unpack_from("%di" % (n + 1), byte_arr, offset=offset)
    offset += SIZE_OF_INT32 * (n + 1)

    # unpack features
    features = []
    for size in sizes[:-1]:
        row = []

        # Now, we need to unpack the feature for multiple statements.
        # The format is: {int n_stage; float feature_vecs[n_stage][vec_len]}
        # where,
        # 1. n_stage is the number of stages
        # 2. feature_vecs is the feature vector for each stage
        # 3. vec_len can be calculated by `(size - 1) / n_stmts`
        if size == 0:
            # failed during lowering
            features.append(np.zeros((1, vec_len)))
        else:
            n_stmts = struct.unpack_from("f", byte_arr, offset=offset)
            n_stmts = int(n_stmts[0] + 0.5)
            if n_stmts == 0:
                # no feature is extracted
                features.append(np.zeros((1, vec_len)))
                continue

            tmp_vec_len = (size - 1) // n_stmts
            offset += SIZE_OF_FLOAT32
            if tmp_vec_len != vec_len:
                vec_len = tmp_vec_len

            assert tmp_vec_len * n_stmts == size - 1
            for _ in range(n_stmts):
                x = struct.unpack_from(
                    "%df" % vec_len, byte_arr, offset=offset)
                offset += vec_len * SIZE_OF_FLOAT32
                row.append(x)

            features.append(np.array(row))

    # unpack normalized_throughputs
    m = sizes[-1]
    normalized_throughputs = struct.unpack_from(
        "%df" % m, byte_arr, offset=offset)
    offset += m * SIZE_OF_FLOAT32

    assert offset == len(byte_arr), "%d vs %d" % (offset, len(byte_arr))
    task_ids = 0
    return np.array(features, dtype=object), np.array(normalized_throughputs), np.array(task_ids)


def get_features_from_stmts(target, stmts, binds, n_skip_cache=0, max_n_buf=5, store_path="./"):
    func = akg.tvm.get_global_func("get_features_from_stmts")
    byte_arr = func(target, stmts, binds, n_skip_cache, max_n_buf, store_path)
    return unpack_feature(byte_arr)[0]
