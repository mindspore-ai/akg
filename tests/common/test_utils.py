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

"""test_utils"""

import math
import random
import numpy as np
import akg.tvm
from akg.utils.validation_check import MAX_DATA_SIZE
from akg.utils.format_transform import get_bytes


def compute_blockdim(shape):
    size = 1
    if isinstance(shape, (list, tuple)):
        for i in shape:
            size = size * i
    elif isinstance(shape, int):
        size = shape
    else:
        size = 2
    return min(32, math.ceil(size / 8192 + 1))


def process_dynamic_shape(shapes, attrs, keep_axis=None):
    dynamic_shape_args = []

    if len(shapes) == 0 or not attrs.get("dynamic"):
        return shapes, dynamic_shape_args

    new_shapes = []
    prefix = "I"

    keep_axis_local = keep_axis

    if isinstance(keep_axis, int):
        keep_axis_local = [keep_axis]

    for shape in shapes:
        dynamic_shape = []
        for i in range(len(shape)):
            if (i in keep_axis_local) or ((i - len(shape)) in keep_axis_local):
                dynamic_shape.append(shape[i])
            else:
                dynamic_shape.append(akg.tvm.var(prefix + str(i)))
                dynamic_shape_args.append(shape[i])

        new_shapes.append(dynamic_shape)
        prefix += "I"

    return new_shapes, dynamic_shape_args


def gen_random_shape(shape_dim, slope=0, min_value=None, max_value=None):
    """
    Generate a list of random integer with length equals shape_dim within range [min_value, max_value];

    Args:
        shape_dim : length of output random shape
        slope : only represents the tendency of random shape's value, not mathematical slope of random shape;
                slope = -1 tend to generate random shape list with largest value at the beginning and smallest value at the end
                slope = 0 tend to generate random shape list with nearly average value among list
                slope = 1 tend to generate random shape list with smallest value at the beginning and largest value at the end
    """
    if shape_dim <= 0:
        raise ValueError("Shape dim should be positive.")

    def _build_limit(limit, default):
        if limit is None:
            limit = default
        res = list()
        nonlocal shape_dim
        if isinstance(limit, (tuple, list)):
            if len(limit) != shape_dim:
                raise ValueError(
                    "Min/Max value should have same length with shape_dim")
            res = limit
        elif isinstance(limit, int):
            res = [limit] * shape_dim
        else:
            raise TypeError(
                "Min/Max value should be int or list of int with same length of shape_dim")
        return res

    device_limit = MAX_DATA_SIZE // get_bytes("float32")
    if max_value is None and shape_dim > 1:
        limit_avg = int(math.pow(device_limit, 1 / shape_dim))

        if slope == 0:
            max_value = [limit_avg] * shape_dim
        else:
            ratio = np.arange(-1/2, 1/2 + 1/shape_dim, 1/shape_dim)
            if len(ratio) > shape_dim:
                new_ratio = list()
                for i, r in enumerate(ratio):
                    if i == len(ratio)//2 - 1:
                        new_ratio.append(0)
                    elif i == len(ratio)//2:
                        continue
                    else:
                        new_ratio.append(r)
                ratio = new_ratio
            if slope == -1:
                ratio.reverse()
            max_value = list()
            for i, r in enumerate(ratio):
                max_value.append(int((1 + ratio[i]) * limit_avg))

    shape_min = _build_limit(min_value, 1)
    shape_extent = _build_limit(max_value, device_limit)
    random_shape = list()
    for mn, mx in zip(shape_min, shape_extent):
        random_shape.append(random.randint(mn, mx))
    return random_shape
