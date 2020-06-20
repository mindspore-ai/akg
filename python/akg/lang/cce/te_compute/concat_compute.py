#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019 Huawei Technologies Co., Ltd
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

"""concat compute"""
import akg.tvm
from .util import dtype_check_decorator


@dtype_check_decorator
def concat(raw_tensors, axis):
    """
    concat shapes at axis,  support int8, uint8, int16, int32 float16, float32

    Args:
        raw_tensors (list[tvm.tensor.Tensor]): list of tensors
        axis (int): concat axis

    Returns:
        concat tensor
    """
    concat_para_check(raw_tensors, axis)

    def _get_input_tensors():
        shapes = []
        for in_tensor in list(raw_tensors):
            shape = [int(in_tensor.shape[i].value) for i in range(len(in_tensor.shape))]
            shapes.append(shape)

        shapes_list = list(shapes)
        return shapes_list

    shapes = _get_input_tensors()

    res_shape = shapes[0][:]
    for i in range(1, len(shapes)):
        res_shape[axis] += shapes[i][axis]

    sel = []
    n_tensor = len(raw_tensors)

    def compute_func(*indice):
        if n_tensor > 1:
            for nn in range(n_tensor - 1):
                if nn == 0:
                    tensor_a = raw_tensors[0]
                    tensor_b = raw_tensors[1]
                    c_shape = shapes[0][:]
                    indice2 = list(indice[:])
                    indice2[axis] = indice[axis] - tensor_a.shape[axis]
                    sel.append(akg.tvm.expr.Select(indice[axis] < c_shape[axis],
                                                   tensor_a[indice], tensor_b[tuple(indice2)]))

                    c_shape[axis] += shapes[1][axis]
                else:
                    tensor_a = sel[nn - 1]
                    tensor_b = raw_tensors[nn + 1]
                    indice2 = list(indice[:])
                    indice2[axis] = indice[axis] - c_shape[axis]
                    sel.append(akg.tvm.expr.Select(indice[axis] < c_shape[axis], tensor_a, tensor_b[tuple(indice2)]))
                    c_shape[axis] += shapes[nn + 1][axis]
        else:
            return raw_tensors[0][indice]

        return sel[-1]

    res = akg.tvm.compute(res_shape, compute_func, name="concat", tag="concat")

    return res


def concat_para_check(raw_tensors, axis):
    """
    concat parameter check

    Args:
        raw_tensors (list[tvm.tensor.Tensor]): list of tensors
        axis (int): concat axis

    Returns:
        rasie runtime error
    """

    # check shape
    if axis < 0 or axis >= len(raw_tensors[0].shape):
        raise RuntimeError("concat axis must be in 0-%d, actual is %d" % (len(raw_tensors[0].shape), axis))

    for i in range(1, len(raw_tensors)):
        if raw_tensors[i].dtype != raw_tensors[0].dtype:
            raise RuntimeError("dtype must be the same to each other")
        for j in range(len(raw_tensors[0].shape)):
            if (j != axis) and (raw_tensors[i].shape[j].value != raw_tensors[0].shape[j].value):
                raise RuntimeError("concat input shape len must be the same to each other except concat axis")
