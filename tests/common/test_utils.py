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

from akg import tvm
import math

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
                dynamic_shape.append(tvm.var(prefix + str(i)))
                dynamic_shape_args.append(shape[i])

        new_shapes.append(dynamic_shape)
        prefix += "I"

    return new_shapes, dynamic_shape_args
