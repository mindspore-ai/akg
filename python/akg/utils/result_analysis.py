#!/usr/bin/env python3
# coding: utf-8
# Copyright 2019-2022 Huawei Technologies Co., Ltd
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

"""result compare function"""
import logging
import json
import numpy as np
import akg.tvm as tvm


def result_compare(actual, bench_mark, r_tol=5e-3):
    """function for compare result."""
    error = 0
    count = 0
    last_err = -2
    continue_err = 0
    max_continue = -1
    max_end = 0
    logging.debug(actual.shape)
    logging.debug(bench_mark.shape)

    actual = actual.reshape((actual.size,))
    len_a = actual.size
    bench_mark = bench_mark.reshape((bench_mark.size,))
    len_b = bench_mark.size
    if len_a != len_b:
        return False

    def _update_error():
        nonlocal error
        nonlocal last_err
        nonlocal count
        nonlocal continue_err
        nonlocal max_continue
        nonlocal max_end
        error += 1
        if last_err + 1 == count:
            continue_err += 1
        else:
            if continue_err > max_continue:
                max_continue = continue_err
                max_end = last_err
            continue_err = 1
        last_err = count

    for i in range(len_a):
        a = actual[i]
        b = bench_mark[i]
        if abs(a - b) > abs(b) * r_tol or np.isnan(a):
            _update_error()
        count += 1
    if continue_err > max_continue:
        max_continue = continue_err
        max_end = last_err
    logging.debug("error num: %d/%d (%.2f%%)", error, count, 100.0 * error / count)
    logging.debug("longest error range: [%d, %d]", max_end - max_continue + 1, max_end)
    if max_continue >= 16:
        return False
    logging.debug("\n\n******************** test ok *****************\n\n")
    return True


def akg_fp16_mean(inputs, axis=None, keepdims=True):
    size = 1
    for dim in axis:
        size = size * inputs.shape[dim]
    expect = np_bisect_sum(inputs, axis=axis, keepdims=keepdims) * np.float16(1 / size)
    return expect


def np_bisect_sum(inputs, axis=None, keepdims=True):
    """numpy bisection summation."""
    shape = inputs.shape
    size = 1
    for dim in axis:
        size = size * shape[dim]
    if size <= 2:
        expect = np_bisect_sum_fp16(inputs, axis=tuple(axis), keepdims=keepdims)
    else:
        expect = np.sum(inputs.astype("float32"), axis=tuple(axis), keepdims=keepdims).astype("float16")
    return expect


def np_bisect_sum_fp16(inputs, axis=None, keepdims=True):
    """
    Function for expected result of bisect sum operation.

    Note:
        For fp16 data, np.sum doesn't have enough accuracy, so use bisect sum instead.
    """
    if axis is None:
        axis = []
    if isinstance(axis, int):
        expect = bisect_sum(inputs, axis, keepdims)
    elif isinstance(axis, (list, tuple)):
        axis = sorted(axis)
        expect = inputs
        i = 0
        for x in axis:
            expect = bisect_sum(expect, x if keepdims else x - i, keepdims)
            i = i + 1
    return expect


def _bisect_sum_axis0(data, reduce_num, tail_num):
    """Bisect sum on axis 0."""
    s = data
    s[0:tail_num, :] = data[0:tail_num, :] + data[reduce_num:reduce_num + tail_num, :]
    while reduce_num != 1:
        s = s[0:reduce_num // 2, :] + s[reduce_num // 2:reduce_num, :]
        reduce_num = reduce_num // 2
    return s


def _bisect_sum_axis1(data, reduce_num, tail_num):
    """Bisect sum on axis 1."""
    s = data
    s[:, 0:tail_num, :] = data[:, 0:tail_num, :] + data[:, reduce_num:reduce_num + tail_num, :]
    while reduce_num != 1:
        s = s[:, 0:reduce_num // 2, :] + s[:, reduce_num // 2:reduce_num, :]
        reduce_num = reduce_num // 2
    return s


def _bisect_sum_axis2(data, reduce_num, tail_num):
    """Bisect sum on axis 2."""
    s = data
    s[:, :, 0:tail_num, :] = data[:, :, 0:tail_num, :] + data[:, :, reduce_num:reduce_num + tail_num, :]
    while reduce_num != 1:
        s = s[:, :, 0:reduce_num // 2, :] + s[:, :, reduce_num // 2:reduce_num, :]
        reduce_num = reduce_num // 2
    return s


def _bisect_sum_axis3(data, reduce_num, tail_num):
    """Bisect sum on axis 3."""
    s = data
    s[:, :, :, 0:tail_num, :] = data[:, :, :, 0:tail_num, :] + data[:, :, :, reduce_num:reduce_num + tail_num, :]
    while reduce_num != 1:
        s = s[:, :, :, 0:reduce_num // 2, :] + s[:, :, :, reduce_num // 2:reduce_num, :]
        reduce_num = reduce_num // 2
    return s


def _bisect_sum_axis4(data, reduce_num, tail_num):
    """Bisect sum on axis 4."""
    s = data
    s[:, :, :, :, 0:tail_num, :] = data[:, :, :, :, 0:tail_num, :] + \
                                   data[:, :, :, :, reduce_num:reduce_num + tail_num, :]
    while reduce_num != 1:
        s = s[:, :, :, :, 0:reduce_num // 2, :] + s[:, :, :, :, reduce_num // 2:reduce_num, :]
        reduce_num = reduce_num // 2
    return s


def _bisect_sum_axis5(data, reduce_num, tail_num):
    """Bisect sum on axis 5."""
    s = data
    s[:, :, :, :, :, 0:tail_num, :] = data[:, :, :, :, :, 0:tail_num, :] + \
                                      data[:, :, :, :, :, reduce_num:reduce_num + tail_num, :]
    while reduce_num != 1:
        s = s[:, :, :, :, :, 0:reduce_num // 2, :] + s[:, :, :, :, :, reduce_num // 2:reduce_num, :]
        reduce_num = reduce_num // 2
    return s


def _bisect_sum_axis6(data, reduce_num, tail_num):
    """Bisect sum on axis 6."""
    s = data
    s[:, :, :, :, :, :, 0:tail_num, :] = data[:, :, :, :, :, :, 0:tail_num, :] + \
                                         data[:, :, :, :, :, :, reduce_num:reduce_num + tail_num, :]
    while reduce_num != 1:
        s = s[:, :, :, :, :, :, 0:reduce_num // 2, :] + s[:, :, :, :, :, :, reduce_num // 2:reduce_num, :]
        reduce_num = reduce_num // 2
    return s


def _bisect_sum_axis7(data, reduce_num, tail_num):
    """Bisect sum on axis 7."""
    s = data
    s[:, :, :, :, :, :, :, 0:tail_num, :] = data[:, :, :, :, :, :, :, 0:tail_num, :] + \
                                            data[:, :, :, :, :, :, :, reduce_num:reduce_num + tail_num, :]
    while reduce_num != 1:
        s = s[:, :, :, :, :, :, :, 0:reduce_num // 2, :] + s[:, :, :, :, :, :, :, reduce_num // 2:reduce_num, :]
        reduce_num = reduce_num // 2
    return s


def _bisect_sum_axis_last(data, reduce_num, tail_num):
    """Bisect sum on last axis."""
    s = data
    s[..., 0:tail_num] = data[..., 0:tail_num] + data[..., reduce_num:reduce_num + tail_num]
    while reduce_num != 1:
        s = s[..., 0:reduce_num // 2] + s[..., reduce_num // 2:reduce_num]
        reduce_num = reduce_num // 2
    return s


def bisect_sum(a, axis=0, keepdims=True):
    """Axis transformations for bisect sum operation."""
    import math
    shape = a.shape
    if not len(shape) <= 8:
        raise AssertionError("the dimension of input cannot be larger than 8!")
    if axis < 0:
        axis = len(shape) + axis
    dimlen = shape[axis]
    reduce_num = int(math.pow(2, int(math.log(dimlen, 2))))
    tail_num = dimlen - reduce_num

    s1 = np.array(a)
    if axis == len(shape) - 1:
        s = _bisect_sum_axis_last(s1, reduce_num, tail_num)
    elif axis == 0:
        s = _bisect_sum_axis0(s1, reduce_num, tail_num)
    elif axis == 1:
        s = _bisect_sum_axis1(s1, reduce_num, tail_num)
    elif axis == 2:
        s = _bisect_sum_axis2(s1, reduce_num, tail_num)
    elif axis == 3:
        s = _bisect_sum_axis3(s1, reduce_num, tail_num)
    elif axis == 4:
        s = _bisect_sum_axis4(s1, reduce_num, tail_num)
    elif axis == 5:
        s = _bisect_sum_axis5(s1, reduce_num, tail_num)
    elif axis == 6:
        s = _bisect_sum_axis6(s1, reduce_num, tail_num)
    elif axis == 7:
        s = _bisect_sum_axis7(s1, reduce_num, tail_num)
    else:
        raise ValueError("axis should be {} or in range[0, 7], but got {}".format(len(shape) - 1, axis))
    if not keepdims:
        s = np.squeeze(s, axis)
    return s


def get_ticks(stat_info):
    """get ticks from statistic info."""
    aic_out_path = "aic_out"
    calog_path = aic_out_path + "/calog"
    ticks_log_file = calog_path + '/core0_instr_popped_log.dump'
    with open(ticks_log_file, "r") as file:
        line = file.readlines()[-2]
        ticks = int(line.split(",")[1].split('tick:')[1])
    stat_info['run_time'] = ticks


def flattened_index_to_real_index(idx, shape):
    index = []
    index_per_dim = idx
    for i in reversed(range(len(shape))):
        dim_index = index_per_dim % shape[i]
        index_per_dim //= shape[i]
        index.append(dim_index)

    index.reverse()
    return index


def count_unequal_element(data_expected, data_actual, rtol, atol):
    """Function for asserting unequal elements in data_actual and data_expected."""
    if not data_expected.shape == data_actual.shape:
        raise AssertionError("'data_expected' and 'data_actual' should have the same shape")
    count = 0
    eps = 1e-10
    if data_expected.dtype == np.bool_ and data_actual.dtype == np.bool_:
        unequal_index = np.where(np.not_equal(data_expected, data_actual))
    else:
        res_elewise = np.isclose(data_actual, data_expected, atol=atol, rtol=rtol, equal_nan=False)
        unequal_index = np.where(np.logical_not(res_elewise))
    if not isinstance(unequal_index, tuple):
        raise ValueError("unequal_index should be tuple but get %s" % type(unequal_index))
    index_len = len(unequal_index)
    unequal_num = unequal_index[0].size
    unequal_actual = data_actual[unequal_index]
    unequal_expected = data_expected[unequal_index]
    log_max_count = 32
    while count < unequal_num:
        if count >= log_max_count:
            break
        index = []
        for i in range(index_len):
            index.append(unequal_index[i][count])
        a = unequal_actual[count]
        b = unequal_expected[count]
        is_bool = isinstance(a, np.bool_) or isinstance(b, np.bool_)
        is_nan = np.isnan(a) or np.isnan(b)
        is_numeric = not (is_bool or is_nan)
        if is_numeric:
            b_1 = b + eps if b == 0.0 else b
            logging.error("%s: Actual[%s] Expected[%s] Ratio[%s]",
                          str(index), str(a), str(b), str(abs(a - b) / abs(b_1)))
        else:
            logging.error("%s: Actual[%s] Expected[%s]", str(index), str(a), str(b))
        count += 1

    if count != 0:
        if unequal_num > log_max_count:
            logging.error("...")
            logging.error("Total %s mismatch detected!!!, Only print %s...", str(unequal_num), str(log_max_count))
        else:
            logging.error("Total %s mismatch detected!!!", str(unequal_num))


def allclose_nparray(data_expected, data_actual, rtol, atol=1e-08):
    """Compare whether arrays are element-wise equal within tolerances."""
    if not np.allclose(data_expected, data_actual, rtol, atol):
        count_unequal_element(data_expected, data_actual, rtol, atol)


class IOInfo(object):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    def __str__(self):
        return "(" + str(self.name) + ", " + str(self.dtype) + ")"

    def __repr__(self):
        return "(" + str(self.name) + ", " + str(self.dtype) + ")"

    def __hash__(self):
        return hash(self.name + self.dtype)

    def __eq__(self, other):
        if not isinstance(other, IOInfo):
            return False
        return self.name == other.name and self.dtype == other.dtype


def _collect_inputs(input_desc):
    """Collect inputs from input_desc."""
    inputs = []
    if not isinstance(input_desc, list):
        return inputs
    for desc in input_desc:
        inputs.append(IOInfo(desc[0]["tensor_name"], desc[0]["data_type"]))
    return inputs


def precision_analyze(desc: dict, tensors):
    exclude_op_list = ["Minimum", "Maximum", "Reshape", "ZerosLike", "Tile", "Select", "Greater",
                       "SelectGT", "SelectLT", "LessEqual", "Less", "EquivFormat", "ExpandDims", "Transpose",
                       "TransData", "BroadcastTo", "Assign"]
    input_tensors = _collect_inputs(desc["input_desc"])

    # Construct graph of current json
    graph = {}
    ops = {}  # recorder the operator that generates the current output
    for op in desc["op_desc"]:
        output = IOInfo(op["output_desc"][0]["tensor_name"], op["output_desc"][0]["data_type"])
        graph[output] = _collect_inputs(op["input_desc"])
        ops[output] = op["name"]

    def _precision_reduce(x: IOInfo):
        if x in input_tensors:
            logging.debug("Skip {}, because it comes from input tensors".format(x))
            return False
        if x in ops and ops.get(x) in exclude_op_list:
            logging.debug("Skip {}, because it comes from [{}] that will not reduce precision".format(x, ops.get(x)))
            return False
        return True

    def _update_stack(node, visited_list, todo_list):
        visited_list.append(node)
        for t in graph.get(node, []):
            if t not in visited_list:
                todo_list.append(t)

    # DFS search for each tensor in tensors to check if they are casted from fp16
    tensors = tensors if isinstance(tensors, (list, tuple)) else [tensors]
    cast_from_fp16 = [False for _ in range(len(tensors))]
    for i, tensor in enumerate(tensors):
        logging.debug("[{}/{}] Analyzing sources of {}...".format(i + 1, len(tensors), tensor))
        visited = []
        stack = [tensor]
        while len(stack) > 0:
            cur_tensor = stack[-1]
            stack.pop()
            # If output comes from a fp16 tensor, and this tensor is produced by operators that will increase
            # the rounding errors(such as Add), then we mark the output cast_from_fp16
            if cur_tensor.dtype == "float16" and _precision_reduce(cur_tensor):
                logging.info("{} --> {}".format(tensor, cur_tensor))
                cast_from_fp16[i] = True
                break
            _update_stack(cur_tensor, visited, stack)
    return cast_from_fp16


def get_compare_tolerance(json_str: str, output_indexes: list):
    desc = json.loads(json_str)
    compare_tolerance = []

    # Collects output tensors of current json based on output_indexes.
    io = []
    for input_desc in desc["input_desc"] if desc["input_desc"] is not None else []:
        io.append(IOInfo(input_desc[0]["tensor_name"], input_desc[0]["data_type"]))
    for output_desc in desc["output_desc"]:
        io.append(IOInfo(output_desc["tensor_name"], output_desc["data_type"]))
    outputs = [io[idx] for idx in output_indexes]

    analyze_indexes = []  # holds the index of float32 outputs
    for i, o in enumerate(outputs):
        if o.dtype == "float16":
            compare_tolerance.append(1e-3)
        else:
            compare_tolerance.append(1e-4)
            if o.dtype == "float32":
                analyze_indexes.append(i)
    if not analyze_indexes:
        return compare_tolerance

    # Check if these float32 outputs are cast from fp16, if so, use 1e-3 rtol and atol when comparing with numpy
    logging.debug("=============== Precision Analyze ===============")
    analyze_outputs = [outputs[idx] for idx in analyze_indexes]
    cast_from_fp16 = precision_analyze(desc, analyze_outputs)
    for i, v in enumerate(cast_from_fp16):
        if v:
            compare_tolerance[analyze_indexes[i]] = 1e-3
            logging.warning("%s will use tolerance %s when comparing with expect.",
                            str(outputs[analyze_indexes[i]]), str(compare_tolerance[analyze_indexes[i]]))
    logging.debug("============= Precision Analyze End =============")

    return compare_tolerance


def target_profiling(mod, *args, target="cuda", repeat_time=1, device_id=0, need_warm_up=True):
    """Do profiling on gpu/cpu for op"""
    ctx = tvm.context(target, device_id)
    if target == "cpu":
        ftimer = mod.time_evaluator(mod.entry_name, ctx, number=repeat_time,
                                    repeat=10, min_repeat_ms=1000,
                                    preproc_name="cache_flush_cpu_non_first_arg")
        if need_warm_up:
            mod.time_evaluator(mod.entry_name, ctx, number=1000)(*args)
    else:
        ftimer = mod.time_evaluator(mod.entry_name, ctx, number=repeat_time)
    tcost = ftimer(*args).mean
    print("{}: exec={} ms/op".format(ctx, tcost * 1000))
    return tcost
