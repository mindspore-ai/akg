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

"""run file for fused_batch_norm"""
import time
import numpy as np
from akg import tvm
from tests.common.tensorio import compare_tensor
from akg.utils import kernel_exec as utils
from akg.ops.nn.ascend import fused_batch_norm
from tests.common.base import get_rtol_atol
from tests.common.gen_random import random_gaussian
from tests.common.test_utils import process_dynamic_shape, compute_blockdim


DYNAMIC_SETDIM_MAP = {
    (32, 128, 7, 7, 16): (1, 4, 7, 7, 16),
    (32, 16, 14, 14, 16): (1, 4, 14, 14, 16),
    (32, 16, 56, 56, 16): (1, 4, 1, 56, 16),
    (32, 32, 28, 28, 16): (1, 1, 28, 28, 16),
    (32, 32, 7, 7, 16): (1, 4, 7, 7, 16),
    (32, 4, 112, 112, 16): (1, 1, 4, 56, 16),
    (32, 4, 56, 56, 16): (1, 4, 1, 56, 16),
    (32, 64, 14, 14, 16): (1, 4, 14, 14, 16),
    (32, 8, 28, 28, 16): (1, 1, 14, 14, 16),
    (32, 8, 56, 56, 16): (1, 4, 1, 56, 16),
    (32, 16, 28, 28, 16): (1, 1, 14, 14, 16),
    (32, 32, 14, 14, 16): (1, 4, 14, 14, 16),
}


def fused_batch_norm_manual_setdim(shape):
    """manual setdim for fused batch norm with dynamic shape"""
    from akg import dim
    info = dim.Dim()
    for i, d in enumerate(DYNAMIC_SETDIM_MAP.get(shape, [])):
        info.setdim(index=0, axis=i, tilel1=d, tilel0=1)
    return str(info)


def fused_batch_norm_run(shape, dtype, momentum, eps, is_training, data_format,
                         axis, kernel_name, attrs):
    """run func for fused_batch_norm, used in test and tuning."""
    if attrs is None:
        attrs = {}
    if attrs.get("dynamic"):
        attrs["dim"] = fused_batch_norm_manual_setdim(shape)
    axis, mean, mean_new, np_beta, np_beta_, \
        np_data, np_data_, np_gamma, np_gamma_, \
        np_mean, np_mean_, np_var, np_var_, \
        _, support_list, var, var_new = gen_input_data(
            axis, data_format, dtype, is_training, momentum, shape)

    build_shape = [np_data.shape]
    build_shape, dynamic_shape_args = process_dynamic_shape(
        build_shape, attrs, -1)

    data_shape = build_shape[0]
    gamma_shape = [1 if i in [0, 2, 3] else data_shape[i]
                   for i in range(len(data_shape))]
    mean_shape = var_shape = beta_shape = gamma_shape
    build_shape += [gamma_shape, beta_shape, mean_shape, var_shape]

    build_attrs = {"momentum": momentum, "eps": eps, "is_training": is_training, "data_format": data_format,
                   "axis": axis}
    if 'tuning' in attrs.keys():
        is_tunning = attrs.get("tuning", False)
        kernel_name = attrs.get("kernel_name", False)
        mod = utils.op_build_test(fused_batch_norm,
                                  [build_shape],
                                  [dtype, dtype, dtype, dtype, dtype],
                                  op_attrs=[build_attrs],
                                  kernel_name=kernel_name,
                                  attrs=attrs, tuning=is_tunning)
        if is_tunning:
            expects, outputs = gen_data(dtype, eps, is_training, mean, mean_new,
                                        np_beta, np_data, np_gamma,
                                        support_list, var, var_new)
            if is_training:
                return mod, expects, {
                    "args": (np_data_, np_gamma_, np_beta_, np_mean_, np_var_,
                             outputs[0], outputs[1], outputs[2],
                             outputs[3], outputs[4]),
                    'outputs': ((-5, -4, -3, 3, 4, -2, -1)),
                    'tuning': False
                }
            else:
                return mod, expects, (np_data_, np_gamma_, np_beta_,
                                      np_mean_, np_var_, outputs[0])
        else:
            return mod
    else:
        expects, outputs = gen_data(dtype, eps, is_training, mean, mean_new,
                                    np_beta, np_data, np_gamma, support_list,
                                    var, var_new)
        attrs["enable_double_buffer"] = False
        mod = utils.op_build_test(fused_batch_norm,
                                  [build_shape], [
                                      dtype, dtype, dtype, dtype, dtype],
                                  op_attrs=[build_attrs],
                                  kernel_name=kernel_name,
                                  attrs=attrs)
        if is_training:
            outputs = utils.mod_launch(mod, (np_data_, np_gamma_, np_beta_,
                                             np_mean_, np_var_,
                                             outputs[0], outputs[1], outputs[2],
                                             outputs[3], outputs[4]),
                                       (-5, -4, -3, 3, 4, -2, -1), expect=expects)
            outputs = [outputs] if len(expects) == 1 else list(outputs)
        else:
            args = [np_data_, np_gamma_, np_beta_,
                    np_mean_, np_var_, outputs[0]]
            if attrs.get("dynamic"):
                args += dynamic_shape_args
                block_dim = compute_blockdim(shape)
                args.append(block_dim)

            outputs = [utils.mod_launch(
                mod, args, outputs=[5, ], expect=expects)]
        rtol, atol = get_rtol_atol("fused_batch_norm", dtype)
        results = list(map(lambda x, y:
                           compare_tensor(x, y, rtol=rtol, atol=atol),
                           outputs, expects))

        results.append([rtol, atol])

        return (np_data_, np_gamma_, np_beta_, np_mean_, np_var_), \
            outputs, expects, results


def gen_data(dtype, eps, is_training, mean, mean_new, np_beta,
             np_data, np_gamma, support_list, var, var_new):
    normalize_data = (np_data - mean) / np.sqrt(var + eps)
    expect_ = np_gamma * normalize_data + np_beta
    fake_mean_new = mean_new
    fake_var_new = var_new
    expects = [expect_, fake_mean_new, fake_var_new,
               mean_new, var_new, mean, var] if is_training else [expect_]
    if dtype != "float32":
        for i in range(len(expects)):
            expects[i] = expects[i].astype(support_list[dtype])
    outputs = [np.full(e.shape, np.nan, dtype) for e in expects]
    return expects, outputs


def gen_input_data(axis, data_format, dtype, is_training, momentum, shape):
    support_t = {"float16": np.float16, "float32": np.float32}
    is_special5d = (data_format == "NC1HWC0")
    if data_format == "NHWC":
        axis = 3
    if is_special5d:
        mv_shape = (1, shape[1], 1, 1, shape[4])
    else:
        mv_shape = (shape[axis],)

    seed_tmp = int(time.time())
    np_data_ = random_gaussian(shape, miu=1, sigma=0.3,
                               seed=seed_tmp).astype(support_t[dtype])
    np_mean_ = random_gaussian(mv_shape, miu=1, sigma=0.1,
                               seed=seed_tmp + 1).astype(support_t[dtype])
    np_var_ = random_gaussian(mv_shape, miu=1, sigma=0.1,
                              seed=seed_tmp + 2).astype(support_t[dtype])
    np_gamma_ = random_gaussian(mv_shape, miu=1, sigma=0.1,
                                seed=seed_tmp + 3).astype(support_t[dtype])
    np_beta_ = random_gaussian(mv_shape, miu=1, sigma=0.1,
                               seed=seed_tmp + 4).astype(support_t[dtype])
    if dtype != "float32":
        np_data = np_data_.astype(np.float32)
        np_mean = np_mean_.astype(np.float32)
        np_var = np_var_.astype(np.float32)
        np_gamma = np_gamma_.astype(np.float32)
        np_beta = np_beta_.astype(np.float32)
    else:
        np_data = np_data_
        np_mean = np_mean_
        np_var = np_var_
        np_gamma = np_gamma_
        np_beta = np_beta_

    mean_new = None
    var_new = None
    if is_training:
        if is_special5d:
            axes = (0, 2, 3)
            if dtype == "float32":
                mean = np.mean(np_data.astype("float64"),
                               axis=axes, keepdims=True).astype("float32")
                var = np.var(np_data.astype("float64"), axis=axes,
                             keepdims=True).astype("float32")
            else:
                mean = np.mean(np_data, axis=axes, keepdims=True)
                var = np.var(np_data, axis=axes, keepdims=True)
        else:
            if axis < 0:
                axis += len(shape)
            axes = tuple([i for i in range(len(shape)) if i != axis])
            if dtype == "float32":
                mean = np.mean(np_data.astype("float64"),
                               axis=axes, keepdims=False).astype("float32")
                var = np.var(np_data.astype("float64"), axis=axes,
                             keepdims=False).astype("float32")
            else:
                mean = np.mean(np_data, axis=axes, keepdims=False)
                var = np.var(np_data, axis=axes, keepdims=False)

        mean_new = momentum * np_mean + (1 - momentum) * mean
        var_new = momentum * np_var + (1 - momentum) * var
    else:
        mean = np_mean
        var = np_var
    return axis, mean, mean_new, np_beta, np_beta_, np_data, np_data_, \
        np_gamma, np_gamma_, np_mean, np_mean_, np_var, np_var_, \
        seed_tmp, support_t, var, var_new
