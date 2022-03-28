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
import akg
import numpy as np
from akg import tvm
from functools import reduce
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.nn.ascend import maxpool, maxpool_with_argmax, maxpool_with_argmax_dynamic
from akg.utils.dsl_create import cal_pad_shapes_by_strategy, get_value
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
import math


def compute_blockdim(shape):
    size = 0
    if isinstance(shape, (list, tuple)):
        for i in shape:
            size = size * i
    elif isinstance(shape, int):
        size = shape
    else:
        size = 2
    return min(32, math.ceil(size / 8192 + 1))


def benchmark(input_0, kernel, stride, pad):
    sh, sw = stride
    n, c1, h, w, c0 = input_0.shape
    kh, kw = kernel

    [ph_h, ph_t, pw_h, pw_t], [out_size_h, out_size_w] = \
        cal_pad_shapes_by_strategy(input_0.shape, kernel, stride, pad)
    out_size_w = get_value(out_size_w, akg.tvm.expr.IntImm)
    out_size_h = get_value(out_size_h, akg.tvm.expr.IntImm)

    out_shape = (n, c1, out_size_h, out_size_w, c0)
    mask_shape = (n, c1, kh, kw, out_size_h, out_size_w, c0)

    min_value = -65504.0 if input_0.dtype == 'float16' \
        else -340282346638528859811704183484516925440.0

    out = np.full(out_shape, min_value, dtype=input_0.dtype)
    mask = np.zeros(mask_shape)

    inputpad = np.full((n, c1, h + ph_h + ph_t, w + pw_h + pw_t, c0),
                       np.finfo(input_0.dtype).min, dtype=input_0.dtype)
    inputpad[:, :, ph_h:ph_h + h, pw_h:pw_h + w, :] = input_0

    for i in range(out_size_h):
        for j in range(out_size_w):
            out[:, :, i, j, :] = \
                np.max(inputpad[:, :, i * sh:i * sh + kh,
                       j * sw:j * sw + kw, :], axis=(2, 3))

    kerneled_shape_tmp = (inputpad.shape[0], inputpad.shape[1],
                          kh * kw, inputpad.shape[4])
    maxid = np.zeros(out_shape)
    for i in range(out_size_h):
        for j in range(out_size_w):
            maxid[:, :, i, j, :] = \
                np.argmax(np.reshape(
                    inputpad[:, :, i * sh:i * sh + kh, j * sw:j * sw + kw, :],
                    kerneled_shape_tmp), axis=2)

    mask_shape_f = [n, c1, kh * kw, out_size_h, out_size_w,  c0]
    mask = np.reshape(mask, tuple(mask_shape_f))

    index_shape = [n, c1, 1, out_size_h, out_size_w, c0]

    def cal_num(shape):
        return reduce(lambda i, j: i * j, [shape[i] for i in range(len(shape))])

    n_indexs = [i for i in range(n) for _ in range(cal_num(index_shape[1:]))]
    c1_indexs = [i for i in range(c1)
                 for _ in range(cal_num(index_shape[2:]))] * n
    ho_indexs = [i for i in range(out_size_h)
                 for _ in range(cal_num(index_shape[4:]))] * \
        cal_num(index_shape[:3])
    wo_indexs = [i for i in range(out_size_w)
                 for _ in range(cal_num(index_shape[5:]))] * \
        cal_num(index_shape[:4])
    c0_indexs = list(range(c0)) * cal_num(index_shape[:-1])

    mask[n_indexs, c1_indexs, maxid.flatten().astype(
        np.int32), ho_indexs, wo_indexs, c0_indexs] = 1
    mask = np.reshape(mask, tuple(mask_shape))

    out = out.astype(input_0.dtype)
    mask = mask.astype(input_0.dtype)
    return out, mask


def maxpool_with_argmax_run(shape, kernel, stride, pad, dsl, dtype, attrs_op=None, polyhedral=True, attrs=None):
    attrs = {} if attrs is None else attrs
    attrs_op = {} if attrs_op is None else attrs_op
    build_shape = []
    arg_list = []
    attrs.update(attrs_op)
    if attrs.get("dynamic"):
        for i in range(len(shape)):
            if i == len(shape) - 1:
                build_shape.append(shape[i])
            else:
                tmp_var = tvm.var("I" + str(i))
                build_shape.append(tmp_var)
                arg_list.append(shape[i])
    else:
        build_shape = shape
    arg_len = len(arg_list)
    if 'tuning' in attrs.keys():
        t = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(maxpool_with_argmax,
                                  [shape], [dtype], op_attrs=[
                                      kernel, stride, pad],
                                  kernel_name=kernel_name, attrs=attrs, tuning=t)
        if t:
            input_0, expects, outputs = \
                gen_data(dtype, kernel, pad, shape, stride)
            return mod, expects, \
                {"args": (input_0, outputs[0], outputs[1]),
                 'outputs': (-2 - arg_len, -1 - arg_len), 'tuning': False}
        else:
            return mod
    else:
        if polyhedral:
            if attrs.get("dynamic") and len(build_shape) > 0:
                mod = utils.op_build_test(maxpool_with_argmax_dynamic,
                                          [build_shape], [dtype], op_attrs=[
                                              kernel, stride, pad],
                                          kernel_name='maxpool', attrs=attrs)
            else:
                mod = utils.op_build_test(maxpool_with_argmax,
                                          [shape], [dtype], op_attrs=[
                                              kernel, stride, pad],
                                          kernel_name='maxpool', attrs=attrs)
        else:
            from akg.ops.nn.ascend.maxpool import maxpool_manual_schedule
            mod = maxpool_manual_schedule(
                shape, kernel, stride, pad, dtype, attrs=attrs, polyhedral=polyhedral)
        input_0, expects, outputs = \
            gen_data(dtype, kernel, pad, shape, stride, attrs)
        args = [input_0, outputs[0], outputs[1]]
        if attrs is not None and attrs.get("dynamic"):
            args = args + arg_list
            block_dim = compute_blockdim(shape)
            args.append(block_dim)
            outputs = utils.mod_launch(
                mod, args, (-3 - arg_len, -2 - arg_len), expect=expects)
        else:
            outputs = utils.mod_launch(
                mod, args, (-2 - arg_len, -1 - arg_len), expect=expects)

        rtol, atol = get_rtol_atol("maxpool", dtype)
        results = list(map(lambda x, y:
                           compare_tensor(x, y, rtol=rtol, atol=atol),
                           outputs, expects))
        return input_0, outputs, expects, all(results)


def gen_data(dtype, kernel, pad, shape, stride, attrs=None):
    support_list = {"float16": np.float16, "float32": np.float32}
    import time
    seed_tmp = int(time.time())
    input_0 = random_gaussian(shape, miu=0,
                            sigma=0.1, seed=seed_tmp).astype(support_list[dtype])
    expect_max, expect_mask = benchmark(input_0, kernel, stride, pad)
    out_shape = expect_max.shape
    mask_shape = expect_mask.shape
    res = np.full(out_shape, -1, dtype)
    res_mask = np.full(mask_shape, -1, dtype)
    if attrs is not None and attrs.get("dynamic"):
        expect_mask = np.full(expect_mask.shape, 0.0, dtype)
    return input_0, [expect_max, expect_mask], [res, res_mask]
