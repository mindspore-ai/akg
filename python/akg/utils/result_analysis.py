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

"""result compare function"""
import logging
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

    for i in range(len_a):
        a = actual[i]
        b = bench_mark[i]
        if abs(a - b) > abs(b) * r_tol:
            error += 1

            if last_err + 1 == count:
                continue_err += 1
            else:
                if continue_err > max_continue:
                    max_continue = continue_err
                    max_end = last_err
                continue_err = 1
            last_err = count

        elif np.isnan(a):
            error += 1

            if last_err + 1 == count:
                continue_err += 1
            else:
                if continue_err > max_continue:
                    max_continue = continue_err
                    max_end = last_err
                continue_err = 1
            last_err = count
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


def bisect_sum(a, axis=0, keepdims=True):
    """Axis transformations for bisect sum operation."""
    import math
    shape = a.shape
    if not len(shape) <= 8:
        raise AssertionError("the dimension of input cannot be larger than 6!")
    if axis < 0:
        axis = len(shape) + axis
    dimlen = shape[axis]
    reduce_num = int(math.pow(2, int(math.log(dimlen, 2))))
    tail_num = dimlen - reduce_num

    s1 = np.array(a)
    s = s1

    if axis == len(shape) - 1:
        s[..., 0:tail_num] = s1[..., 0:tail_num] + s1[..., reduce_num:reduce_num + tail_num]
        while reduce_num != 1:
            s = s[..., 0:reduce_num // 2] + s[..., reduce_num // 2:reduce_num]
            reduce_num = reduce_num // 2
    elif axis == 0:
        s[0:tail_num, :] = s1[0:tail_num, :] + s1[reduce_num:reduce_num + tail_num, :]
        while reduce_num != 1:
            s = s[0:reduce_num // 2, :] + s[reduce_num // 2:reduce_num, :]
            reduce_num = reduce_num // 2
    elif axis == 1:
        s[:, 0:tail_num, :] = s1[:, 0:tail_num, :] + s1[:, reduce_num:reduce_num + tail_num, :]
        while reduce_num != 1:
            s = s[:, 0:reduce_num // 2, :] + s[:, reduce_num // 2:reduce_num, :]
            reduce_num = reduce_num // 2
    elif axis == 2:
        s[:, :, 0:tail_num, :] = s1[:, :, 0:tail_num, :] + s1[:, :, reduce_num:reduce_num + tail_num, :]
        while reduce_num != 1:
            s = s[:, :, 0:reduce_num // 2, :] + s[:, :, reduce_num // 2:reduce_num, :]
            reduce_num = reduce_num // 2
    elif axis == 3:
        s[:, :, :, 0:tail_num, :] = s1[:, :, :, 0:tail_num, :] + s1[:, :, :, reduce_num:reduce_num + tail_num, :]
        while reduce_num != 1:
            s = s[:, :, :, 0:reduce_num // 2, :] + s[:, :, :, reduce_num // 2:reduce_num, :]
            reduce_num = reduce_num // 2
    elif axis == 4:
        s[:, :, :, :, 0:tail_num, :] = s1[:, :, :, :, 0:tail_num, :] + \
            s1[:, :, :, :, reduce_num:reduce_num + tail_num, :]
        while reduce_num != 1:
            s = s[:, :, :, :, 0:reduce_num // 2, :] + s[:, :, :, :, reduce_num // 2:reduce_num, :]
            reduce_num = reduce_num // 2
    elif axis == 5:
        s[:, :, :, :, :, 0:tail_num, :] = s1[:, :, :, :, :, 0:tail_num, :] +\
            s1[:, :, :, :, :, reduce_num:reduce_num + tail_num, :]
        while reduce_num != 1:
            s = s[:, :, :, :, :, 0:reduce_num // 2, :] + s[:, :, :, :, :, reduce_num // 2:reduce_num, :]
            reduce_num = reduce_num // 2
    elif axis == 6:
        s[:, :, :, :, :, :, 0:tail_num, :] = s1[:, :, :, :, :, :, 0:tail_num, :] + \
            s1[:, :, :, :, :, :, reduce_num:reduce_num + tail_num, :]
        while reduce_num != 1:
            s = s[:, :, :, :, :, :, 0:reduce_num // 2, :] + s[:, :, :, :, :, :, reduce_num // 2:reduce_num, :]
            reduce_num = reduce_num // 2
    elif axis == 7:
        s[:, :, :, :, :, :, :, 0:tail_num, :] = s1[:, :, :, :, :, :, :, 0:tail_num, :] + \
            s1[:, :, :, :, :, :, :, reduce_num:reduce_num + tail_num, :]
        while reduce_num != 1:
            s = s[:, :, :, :, :, :, :, 0:reduce_num // 2, :] + s[:, :, :, :, :, :, :, reduce_num // 2:reduce_num, :]
            reduce_num = reduce_num // 2
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
    list_a = data_expected.flatten()
    list_b = data_actual.flatten()
    count = 0
    eps = 1e-10
    all_printed = True
    for i, aa in enumerate(list_a):
        a = list_b[i]
        b = aa
        is_bool = isinstance(a, np.bool_) or isinstance(b, np.bool_)
        is_nan = np.isnan(a) or np.isnan(b)
        is_numeric = not (is_bool or is_nan)
        if (is_bool and a != b) or (is_numeric and abs(a - b) > (atol + rtol * abs(b))) or is_nan:
            if count < 100:
                index = flattened_index_to_real_index(i, data_expected.shape)
                if is_numeric:
                    b_1 = b + eps if b == 0.0 else b
                    logging.error("%s: Actual[%s] Expected[%s] Ratio[%s]",
                                  str(index), str(a), str(b), str(abs(a - b) / abs(b_1)))
                else:
                    logging.error("%s: Actual[%s] Expected[%s]", str(index), str(a), str(b))
            else:
                all_printed = False
            count += 1

    if count != 0:
        if not all_printed:
            logging.error("...")
            logging.error("Total %s mismatch detected!!!, Only print 100...", str(count))
        else:
            logging.error("Total %s mismatch detected!!!", str(count))

    if not count <= int(len(list_a)):
        raise AssertionError


def allclose_nparray(data_expected, data_actual, rtol, atol=1e-08):
    """Compare whether arrays are element-wise equal within tolerances."""
    if not np.allclose(data_expected, data_actual, rtol, atol):
        count_unequal_element(data_expected, data_actual, rtol, atol)

def gpu_profiling(mod, *args, repeat_time=1, device_id=0):
    """Do profiling on gpu for cuda op"""
    ctx = tvm.context("cuda", device_id)
    ftimer = mod.time_evaluator(mod.entry_name, ctx, number=repeat_time)
    tcost = ftimer(*args).mean
    print("{}: exec={} sec/op".format(ctx, tcost))
