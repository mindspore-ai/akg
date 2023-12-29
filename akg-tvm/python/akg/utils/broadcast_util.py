#!/usr/bin/env python3
# coding: utf-8
# Copyright 2020 Huawei Technologies Co., Ltd
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

"""broadcast_util"""
from functools import reduce
import akg.tvm
from akg.utils.format_transform import get_shape
from akg.ms.utils import NC1HWC0


def need_broadcast(main_data_shape, main_logical_shape, with_shape):
    """return False if main_data needn't to do broadcast"""

    if not with_shape:
        return False

    if not main_logical_shape:
        return False

    with_data_num = reduce(lambda x, y: x * y, with_shape)
    if with_data_num == 1:
        return False

    if main_logical_shape == with_shape:
        return False

    main_logical_shape_new = main_logical_shape if main_logical_shape else (1,)

    # No special broadcast is needed if there is no pad in data
    if main_logical_shape_new == main_data_shape:
        return False

    if len(main_logical_shape) >= len(with_shape):
        for i in range(0 - len(with_shape), 0):
            if main_logical_shape[i] < with_shape[i]:
                return True
        return False

    return True


def broadcast_by_format(ori_data, logical_shape, format_in, with_shape):
    """
    Do special broadcast for special formats when padding axis needs to broadcast, such as C in NCHW(NC1HWC0).

    Rewrite padding value to broadcast value in special case, for example: op1 * op2, where op1 and op2 are both
    NC1HWC0, and their logical 4D shapes are (4, 1, 3, 3) and (4, 4, 3, 3). op1's shape become (4, 1, 3, 3, 16) after
    transformation from 4D to NC1HWC0. we need to fill the data of axis C0 with broadcast value but not padding value.

    Note:
        There is no need to do broadcast for scalar and DefaultFormat(or NHWC) here.
    """

    ori_data_shape = tuple(get_shape(ori_data))

    if not need_broadcast(ori_data_shape, tuple(logical_shape),
                          tuple(with_shape)):
        return ori_data

    nchw_shape_len = fracz_shape_len = 4
    nc1hwc0_shape_len = 5
    logical_shape_new = tuple(logical_shape) if logical_shape else (1,)
    data_num = reduce(lambda x, y: x * y, logical_shape_new)
    if data_num == 1:
        # this is a scalar
        if len(ori_data_shape) == fracz_shape_len:
            new_data = akg.tvm.compute((1,), lambda i: ori_data[0, 0, 0, i])
        elif len(ori_data_shape) == nc1hwc0_shape_len:
            new_data = akg.tvm.compute((1,), lambda i: ori_data[0, 0, 0, 0, i])
        else:
            raise RuntimeError("Unsupported shape {}".format(ori_data_shape))
        return new_data

    # NC1HWC0
    if format_in == NC1HWC0:
        if len(with_shape) != nchw_shape_len:
            raise ValueError("with_shape must be 4D, while it is {}".format(with_shape))

        # rewrite padding value to broadcast value only if C(NCHW, NHWC is not considered) is the broadcast axis
        if logical_shape[1] == 1:
            new_data = akg.tvm.compute(ori_data_shape, lambda n, c1, h, w, c0: ori_data[n, c1, h, w, 0])
            return new_data

        return ori_data

    raise RuntimeError("Broadcast is unsupported when logical_shape is {}, and format is {}".
                       format(logical_shape, format_in))
