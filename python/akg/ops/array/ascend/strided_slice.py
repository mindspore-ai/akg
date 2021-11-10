#!/usr/bin/env python3
# coding: utf-8
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

"""operator dsl function: strided_slice"""
import copy
import numpy as np
import akg.topi
import akg.tvm
import akg.utils as  utils

def check_args(begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    """check args."""
    if len(begin) != len(end):
        raise Exception("len(begin) is {}, len(end) is {}. They must be identical!".format(len(begin), len(end)))
    if strides is not None:
        if len(begin) != len(strides):
            raise Exception("len(begin) is {}, len(strides) is {}. They must be identical!".
                            format(len(begin), len(strides)))
    for s in strides:
        if s == 0:
            raise Exception("Value in strides[{}] must not be 0!".format(strides))

    if begin_mask < 0 or begin_mask >= (2 ** len(begin)):
        raise Exception("Illegal begin_mask[{}]".format(begin_mask))

    if end_mask < 0 or end_mask >= (2 ** len(begin)):
        raise Exception("Illegal end_mask[{}]".format(end_mask))

    if ellipsis_mask < 0 or ellipsis_mask >= (2 ** len(begin)):
        raise Exception("Illegal ellipsis_mask[{}]".format(ellipsis_mask))

    if ellipsis_mask != 0:  # ellipsis_mask must be a power of two (only one ellipsis)
        if ellipsis_mask & (ellipsis_mask - 1) != 0:
            raise Exception("ellipsis_mask[{}] is not power of two (only one ellipsis).".format(ellipsis_mask))

    if new_axis_mask < 0 or new_axis_mask >= (2 ** len(begin)):
        raise Exception("Illegal new_axis_mask[{}]".format(new_axis_mask))

    if shrink_axis_mask < 0 or shrink_axis_mask >= (2 ** len(begin)):
        raise Exception("Illegal shrink_axis_mask[{}]".format(shrink_axis_mask))

def args_to_slices(begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    """args to slice."""
    check_args(begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    slices = []
    for dim, bgn in enumerate(begin):
        if (ellipsis_mask >> dim) & 1:
            slices.append(Ellipsis)
        elif (new_axis_mask >> dim) & 1:
            slices.append(np.newaxis)
        elif (shrink_axis_mask >> dim) & 1:
            slices.append(bgn)
        else:
            start = None if (begin_mask >> dim) & 1 else bgn
            stop = None if (end_mask >> dim) & 1 else end[dim]
            step = strides[dim]
            slices.append(slice(start, stop, step))
    return slices


def slices_to_args(slices=()):
    """slice to args."""
    begin = []
    end = []
    strides = []
    begin_mask = 0
    end_mask = 0
    ellipsis_mask = 0
    new_axis_mask = 0
    shrink_axis_mask = 0
    for i, arg in enumerate(slices):
        if isinstance(arg, slice):
            begin.append(0 if arg.start is None else arg.start)
            if arg.start is None:
                begin_mask |= 1 << i
            end.append(0 if arg.stop is None else arg.stop)
            if arg.stop is None:
                end_mask |= 1 << i
            strides.append(1 if arg.step is None else arg.step)
        elif arg is np.newaxis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            new_axis_mask |= 1 << i
        elif arg is Ellipsis:
            begin.append(0)
            end.append(0)
            strides.append(1)
            ellipsis_mask |= 1 << i
        elif isinstance(arg, int):
            begin.append(arg)
            end.append(arg + 1)
            strides.append(1)
            shrink_axis_mask |= 1 << i
        else:
            raise Exception("arg ", arg, ' is invalid')
    return begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask


def complete_args(inputs_shape, begin, end, strides, begin_mask,
                  end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask):
    """complete args."""
    check_args(begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)

    # step0: deep copy begin, end, strides
    begin = copy.copy(begin)
    end = copy.copy(end)
    strides = copy.copy(strides)

    # step1: store all bits and calculate new_axis_count
    check_args(begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask)
    begin_list = [(begin_mask >> dim) & 1 for dim in range(len(begin))]
    end_list = [(end_mask >> dim) & 1 for dim in range(len(begin))]
    ellipsis_list = [(ellipsis_mask >> dim) & 1 for dim in range(len(begin))]
    new_axis_list = [(new_axis_mask >> dim) & 1 for dim in range(len(begin))]
    new_axis_count = len([dim for dim in range(len(begin)) if (new_axis_mask >> dim) & 1])
    shrink_list = [(shrink_axis_mask >> dim) & 1 for dim in range(len(begin))]

    # step2: fill the ellipsis using ellipsis_list
    ellipsis_idx = None
    for idx, x in enumerate(ellipsis_list):
        if x:
            ellipsis_idx = idx
            break
    if ellipsis_idx is not None:
        ellipsis_length = len(inputs_shape) - (len(begin) - 1 - new_axis_count)
        idx = ellipsis_idx

        begin.pop(idx)
        end.pop(idx)
        strides.pop(idx)
        begin_list.pop(idx)
        end_list.pop(idx)
        ellipsis_list.pop(idx)
        new_axis_list.pop(idx)
        shrink_list.pop(idx)

        for _ in range(ellipsis_length):
            begin.insert(idx, None)
            end.insert(idx, None)
            strides.insert(idx, 1)
            begin_list.insert(idx, 1)
            end_list.insert(idx, 1)
            ellipsis_list.insert(idx, 0)
            new_axis_list.insert(idx, 0)
            shrink_list.insert(idx, 0)

    # step3: remove new_axis using new_axis_list
    new_axis_index = [idx for idx, x in enumerate(new_axis_list) if x]
    for idx in new_axis_index[::-1]:
        begin.pop(idx)
        end.pop(idx)
        strides.pop(idx)
        begin_list.pop(idx)
        end_list.pop(idx)
        ellipsis_list.pop(idx)
        shrink_list.pop(idx)
        new_axis_list.pop(idx)

    # step4: update (begin, end, strides) using (shrink_list, begin_list, end_list)
    for dim, bgn in enumerate(begin):
        if shrink_list[dim]:
            end[dim] = bgn + 1
            strides[dim] = 1
            continue
        if begin_list[dim]:
            begin[dim] = 0
        if end_list[dim]:
            end[dim] = inputs_shape[dim]

    return begin, end, strides, new_axis_index, shrink_list

@utils.check_input_type(akg.tvm.tensor.Tensor, ((list, tuple), int), ((list, tuple), int),
                          ((list, tuple), int), int, int, int, int, int, (str, type(None)))
def StridedSlice(inputs, begin, end, strides, begin_mask, end_mask, ellipsis_mask, new_axis_mask, shrink_axis_mask, target=utils.CCE):
    """
    Generate an array by slicing input tensor

    Args:
        inputs (tvm.tensor.Tensor): Tensor of type float16, float32.
        begin (Union[list, tuple, int]): The start indexes for slicing.
        end (Union[list, tuple, int]): The end indexes for slicing.
        strides (Union[list, tuple, int]): The strides for slicing.
        begin_mask (int): int32 mask for begin indexes.
        end_mask (int): int32 mask for end indexes.
        ellipsis_mask (int): int32 mask for inserting unspecified dimensions.
        new_axis_mask (int): int32 mask for new dim with length 1.
        shrink_axis_mask (int): int32 mask for shrinking the dims.
    Returns:
        tvm.tensor.Tensor, with the same dtype as inputs.
    
    Supported Platforms:
        'Ascend'
    """

    shape = [x.value for x in inputs.shape]

    # step0~4: complete begin, end, strides
    begin, end, strides, new_axis_index, shrink_list = complete_args(shape, begin, end, strides,
                                                                     begin_mask, end_mask, ellipsis_mask,
                                                                     new_axis_mask, shrink_axis_mask)
    # step5: use topi to do strided_slice using begin, end, strides

    if (shape == [1] and begin == end):
        return akg.tvm.compute(shape, lambda *i: inputs(*i), name="out")

    if inputs.dtype == "uint8":
        inputs_cast = akg.topi.cast(inputs, "int8")
    else:
        inputs_cast = inputs

    out1 = akg.topi.strided_slice(inputs_cast, begin, end, strides)

    # step6: increase out_tensor's dim using new_axis_index
    new_shape = list(out1.shape)
    for idx in new_axis_index[::-1]:
        new_shape.insert(idx, 1)

    # step7: decrease out_tensor's dim using shrink_list
    for idx in new_axis_index[::-1]:
        shrink_list.insert(idx, 0)
    shrink_axis_index = [idx for idx, x in enumerate(shrink_list) if x]
    for idx in shrink_axis_index[::-1]:
        new_shape.pop(idx)

    # step8: reshape out_tensor
    out2 = akg.topi.reshape(out1, tuple(new_shape))

    if inputs.dtype == "uint8":
        out2 = akg.topi.cast(out2, "uint8")

    return out2
