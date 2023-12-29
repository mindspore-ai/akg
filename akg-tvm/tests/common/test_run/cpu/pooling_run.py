# Copyright 2022 Huawei Technologies Co., Ltd
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
# limitations under the License
from akg.ops.nn.cpu import pooling
import akg
import tvm
import topi
import numpy as np
from akg.topi.util import get_const_tuple
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_random import random_gaussian

support_list = {"float32": np.float32}
support_layout_format = {"NCHW", "NCHWc"}


def gen_data(shape_data, kernel, stride, padding, pool_type, dtype, ceil_mode, count_include_pad, data_layout):
    kw, kh = kernel
    sw, sh = stride
    pt, pl, pb, pr = padding
    if data_layout == "NCHW":
        n, ic, ih, iw = shape_data

        input0 = tvm.placeholder((n, ic, ih, iw), name='input0')
        output0 = topi.nn.pool(input0, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                         pool_type=pool_type, ceil_mode=ceil_mode,
                         layout="NCHW", count_include_pad=count_include_pad)

        a_np = random_gaussian((n, ic, ih, iw), miu=1, sigma=0.1).astype(support_list[dtype])
        pad_np = np.zeros(shape=(n, ic, ih+pt+pb, iw+pl+pr)).astype(dtype)
        no_zero = (range(n), range(ic), (range(pt, ih+pt)), (range(pl, iw+pl)))
        pad_np[np.ix_(*no_zero)] = a_np
        _, oc, oh, ow = get_const_tuple(output0.shape)
        b_np = np.zeros(shape=(n, oc, oh, ow)).astype(dtype)

        if pool_type == 'avg':
            for i in range(oh):
                for j in range(ow):
                    if count_include_pad:
                        b_np[:, :, i, j] = np.mean(
                            pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2, 3))
                    else:
                        pad_count = np.sum(
                            pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw] > 0, axis=(2, 3))
                        b_np[:, :, i, j] = np.sum(
                            pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2, 3)) / np.maximum(pad_count, 1)

        elif pool_type == 'max':
            for i in range(oh):
                for j in range(ow):
                    b_np[:, :, i, j] = np.max(
                        pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw], axis=(2, 3))
        output_np = np.zeros(shape=(n, oc, oh, ow)).astype(dtype)
        return a_np, output_np, b_np

    elif data_layout == "NHWC":
        raise ValueError("Only layout NCHWc/NCHW supported on python dsl")
    else:
        # NCHWc
        n, ic_out, ih, iw, ic_in = shape_data

        input0 = tvm.placeholder((n, ic_out, ih, iw, ic_in), name='input0')
        output0 = topi.nn.pool(input0, kernel=[kh, kw], stride=[sh, sw], padding=padding,
                         pool_type=pool_type, ceil_mode=ceil_mode,
                         layout="NCHWc", count_include_pad=count_include_pad)

        a_np = random_gaussian((n, ic_out, ih, iw, ic_in), miu=1, sigma=0.1).astype(support_list[dtype])
        pad_np = np.zeros(shape=(n, ic_out, ih+pt+pb,
                          iw+pl+pr, ic_in)).astype(dtype)
        no_zero = (range(n), range(ic_out), (range(pt, ih+pt)),
                   (range(pl, iw+pl)), range(ic_in))
        pad_np[np.ix_(*no_zero)] = a_np
        _, oc_out, oh, ow, oc_in = get_const_tuple(output0.shape)
        b_np = np.zeros(shape=(n, oc_out, oh, ow, oc_in)).astype(dtype)

        if pool_type == 'avg':
            for i in range(oh):
                for j in range(ow):
                    if count_include_pad:
                        b_np[:, :, i, j, :] = np.mean(
                            pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(2, 3))
                    else:
                        pad_count = np.sum(
                            pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw, :] > 0, axis=(2, 3))
                        b_np[:, :, i, j, :] = np.sum(
                            pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(2, 3)) / np.maximum(pad_count, 1)

        elif pool_type == 'max':
            for i in range(oh):
                for j in range(ow):
                    b_np[:, :, i, j, :] = np.max(
                        pad_np[:, :, i*sh:i*sh+kh, j*sw:j*sw+kw, :], axis=(2, 3))
        output_np = np.zeros(shape=(n, oc_out, oh, ow, oc_in)).astype(dtype)
        return a_np, output_np, b_np


def pooling_run(shape_data, kernel, stride, padding, pool_type, dtype,
                ceil_mode, count_include_pad=True,
                data_layout="NCHWc", poly_sch=True, attrs=None):

    default_attrs = {"enable_auto_fuse": False,"polytops_parameter_shifting": True, "polytops_enable_skewing": False}
    attrs = {} if attrs == None else attrs
    attrs.update(default_attrs)
    attrs["target"] = attrs.get("target", "llvm")
    op_attrs = [kernel, stride, padding, pool_type,
                ceil_mode, count_include_pad, data_layout]

    mod = utils.op_build_test(pooling, (shape_data,), (dtype,),
                              op_attrs=op_attrs, attrs=attrs,
                              kernel_name="pooling_" + pool_type + "_auto", polyhedral=poly_sch)

    data, output, expect = gen_data(
        shape_data, kernel, stride, padding, pool_type, dtype, ceil_mode, count_include_pad, data_layout)
    args = (data, output)
    output = utils.mod_launch(mod, args, expect=expect)
    rtol = 1e-3 if dtype == "float16" else 1e-4
    atol = 1e-3 if dtype == "float16" else 1e-4
    res = np.allclose(output, expect, rtol=rtol, atol=atol)
    print("Test {}".format("Pass" if res else "Fail"))
    target_name = attrs["target"].split()[0]
    if not res:
        mod_source = mod
        if target_name != "llvm":
            mod_source = mod.imported_modules[0]
        print("Error {}:========================".format(target_name))
        print(mod_source.get_source())
        raise AssertionError("Test fail")

    if attrs.get("profiling", False):
        data, output = to_tvm_nd_array(
            [data, output], akg.tvm.context(target_name, 0))
        target_profiling(mod, data, output,
                         target=target_name, repeat_time=attrs.get("repeat_times", 1000))
    return (data, ), output, expect, res
