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

"""segment compute"""
import akg.tvm
from .util import dtype_check_decorator, shape_to_list
from .elewise_compute import binary_elewise_op
from .broadcast_compute import broadcast


@dtype_check_decorator
def unsorted_segment_sum(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment sum, return a new tensor which is the sum along segments of a tensor. only support float16, int32

    Args:
        tensor (tvm.tensor.Tensor): Input tensor
        segment_ids (list): Index of each segment
        num_segments (tvm.tensor.Tensor): Number of distinct segment ids
        init_value (int): Initial value

    Returns:
        tvm.tensor.Tensor, segment_sum(tensor , segment_ids)
    """
    return segment_op(tensor, segment_ids, num_segments, init_value, tensor.dtype, "segment_sum")


@dtype_check_decorator
def unsorted_segment_mean(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment mean, return a new tensor which is the mean along segments of a tensor.only support float16, int32

    Args:
        tensor (tvm.tensor.Tensor): Input tensor
        segment_ids (list): index of each segment
        num_segments (tvm.tensor.Tensor): Number of distinct segment ids
        init_value (int): Initial value

    Returns:
        tvm.tensor.Tensor, segment_mean(tensor , segment_ids)
    """
    return segment_op(tensor, segment_ids, num_segments, init_value, tensor.dtype, "segment_mean")


@dtype_check_decorator
def unsorted_segment_prod(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment prod, return a new tensor which is the prod along segments of a tensor,only support f16, s32

    Args:
        tensor (tvm.tensor.Tensor): Input tensor
        segment_ids (list): Index of each segment
        num_segments (tvm.tensor.Tensor): Number of distinct segment ids
        init_value (int): Initial value

    Returns:
        tvm.tensor.Tensor, segment_prod(tensor , segment_ids)
    """
    return segment_op(tensor, segment_ids, num_segments, init_value, tensor.dtype, "segment_prod")


@dtype_check_decorator
def unsorted_segment_min(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment min, return a new tensor which is the min along segments of a tensor. only support float16, int32.

    Args:
        tensor (tvm.tensor.Tensor): Input tensor
        segment_ids (list): Index of each segment.
        num_segments (tvm.tensor.Tensor): Number of distinct segment ids.
        init_value (int): Initial value.

    Returns:
        tvm.tensor.Tensor, segment_min(tensor , segment_ids).
    """
    return segment_op(tensor, segment_ids, num_segments, init_value, tensor.dtype, "segment_min")


@dtype_check_decorator
def unsorted_segment_max(tensor, segment_ids, num_segments, init_value=0):
    """
    calculate segment max, return a new tensor which is the max along segments of a tensor. only support float16, int32.

    Args:
        tensor (tvm.tensor.Tensor): Input tensor.
        segment_ids (list): Index of each segment.
        num_segments (tvm.tensor.Tensor): Number of distinct segment ids.
        init_value (int): Initial value.

    Returns:
        tvm.tensor.Tensor, segment_max(tensor , segment_ids).
    """
    return segment_op(tensor, segment_ids, num_segments, init_value, tensor.dtype, "segment_max")


def segment_op(tensor, segment_ids, num_segments, init_value, output_dtype, op):
    """factory method of segment operations"""

    def segment_compute(indices):
        """compute_func of unsorted segment mean arithmetic operator"""
        unique_id = []
        for i in segment_ids:
            if i not in unique_id:
                unique_id.append(i)

        def compute_outer_dim(i):
            new_segment_id = list(segment_ids)[:]
            if i in unique_id:
                idx = new_segment_id.index(i)
                new_segment_id[idx] = -1
                tmp = tensor[(idx,) + indices[1:]].astype(output_dtype)
                for _ in range(segment_ids.count(i) - 1):
                    new_segment_id[idx] = -1
                    idx = new_segment_id.index(i)
                    if op in ("segment_sum", "segment_mean"):
                        tmp = tensor[(idx,) + indices[1:]].astype(output_dtype) + tmp
                    elif op == "segment_prod":
                        tmp = tensor[(idx,) + indices[1:]].astype(output_dtype) * tmp
                    elif op == "segment_min":
                        tmp = akg.tvm.min(tensor[(idx,) + indices[1:]].astype(output_dtype), tmp)
                    elif op == "segment_max":
                        tmp = akg.tvm.max(tensor[(idx,) + indices[1:]].astype(output_dtype), tmp)
                    else:
                        raise RuntimeError("operation %s not support yet" % op)
                if op == "segment_mean":
                    tmp = tmp // akg.tvm.const(segment_ids.count(i), output_dtype)
            else:
                tmp = akg.tvm.const(init_value, tensor.dtype)
            return tmp

        res = compute_outer_dim(0)
        for i in range(num_segments)[1:]:
            res = akg.tvm.select(indices[0] == i, compute_outer_dim(i), res)
        return res

    shape = shape_to_list(tensor.shape)

    # check
    if len(segment_ids) > shape[0]:
        raise RuntimeError("the rank of segment_ids should be equal to"
                           "the rank of data's first dimension")
    if max(segment_ids) > num_segments:
        raise RuntimeError("num_segments must be larger than max value of segment_ids,"
                           "while num_segments is %d and max value of segment_ids is %d"
                           % (num_segments, max(segment_ids)))

    name = "data_" + op.split("_")[-2] + '_' + tensor.name.split("_")[-1]

    if max(segment_ids) < 0:
        spec_dtype_list = ["int8", "uint8"]

        output_shape = [num_segments] + shape[1:]
        init_value_const = akg.tvm.const(init_value, output_dtype)

        init_value_tmp = broadcast(init_value_const, output_shape)
        input_tmp_zero = binary_elewise_op(tensor, tensor, "elewise_binary_sub")
        with akg.tvm.tag_scope("broadcast_for_tensor"):
            output_tmp_zero = akg.tvm.compute(output_shape, lambda *indices: input_tmp_zero[(0,) + indices[1:]],
                                              name="output_tmp")

        if output_dtype in spec_dtype_list:
            with akg.tvm.tag_scope("segment_elewise_special"):
                tmp = akg.tvm.compute(output_shape, lambda *indices: output_tmp_zero[indices] + init_value_tmp[indices],
                                      name=name)
        else:
            tmp = binary_elewise_op(output_tmp_zero, init_value_tmp, "elewise_binary_add")
    else:
        lambda_func = lambda *indices: segment_compute(indices)
        shape[0] = num_segments
        str_segment_ids = ",".join([str(i) for i in segment_ids])
        with akg.tvm.tag_scope(op + "|" + str_segment_ids + "|" + str(num_segments) + "|" + str(init_value)):
            tmp = akg.tvm.compute(shape, lambda_func, name=name)

    return tmp
