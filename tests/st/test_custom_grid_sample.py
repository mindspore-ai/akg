# Copyright 2025 Huawei Technologies Co., Ltd
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
# ============================================================================
import os
import time
import numpy as np
import pytest
from functools import wraps

import ms_custom_ops
import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore.common.api import jit
from mindspore import Tensor, mint, nn, ops, context, Profiler
from mindspore.profiler import ProfilerLevel, ProfilerActivity, AicoreMetrics
# from mindspore.common.np_dtype import bfloat16
from mindspore._c_expression import MSContext

def jit_for_graph_mode(fn):
    """
    A decorator that conditionally applies jit to a function at runtime based on the context mode.
    """
    jitted_fn = jit(fn)
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if context.get_context("mode") == context.GRAPH_MODE:
            return jitted_fn(*args, **kwargs)
        return fn(*args, **kwargs)
    return wrapper


def bilinear_interpolate(input_tensor, x, y, H, W):
    """双线性插值"""
    x1 = int(np.floor(x))
    y1 = int(np.floor(y))
    x2 = min(x1 + 1, W - 1)
    y2 = min(y1 + 1, H - 1)

    # 边界检查
    if x1 < 0 or x1 >= W or y1 < 0 or y1 >= H:
        return np.zeros(input_tensor.shape[0], dtype=np.float32)

    # 计算权重
    wx = x - x1
    wy = y - y1

    # 双线性插值
    result = (input_tensor[y1, x1, :] * (1 - wx) * (1 - wy) +
              input_tensor[y1, x2, :] * wx * (1 - wy) +
              input_tensor[y2, x1, :] * (1 - wx) * wy +
              input_tensor[y2, x2, :] * wx * wy)
    return result


def golden_grid_sample(input_data, grid_data, align_corners, padding_mode, interpolation_mode):
    n, h_in, w_in, c = input_data.shape
    N, H_out, W_out, _ = grid_data.shape
    output_data = np.zeros([N, H_out, W_out, c]).astype(np.float32)

    for n in range(N):
        for h in range(H_out):
            for w in range(W_out):
                # 获取归一化坐标
                x_norm = grid_data[n, h, w, 0]
                y_norm = grid_data[n, h, w, 1]

                # 映射到input坐标空间
                if align_corners:
                    x = (x_norm + 1) * (w_in - 1) / 2
                    y = (y_norm + 1) * (h_in - 1) / 2
                else:
                    x = (x_norm + 1) * w_in / 2 - 0.5
                    y = (y_norm + 1) * h_in / 2 - 0.5

                # 边界处理
                if padding_mode == "zeros":  # zeros
                    if x < 0 or x >= w_in or y < 0 or y >= h_in:
                        output_data[n, h, w, :] = 0
                        continue
                elif padding_mode == "border":  # border
                    x = np.clip(x, 0, w_in - 1)
                    y = np.clip(y, 0, h_in - 1)

                # 插值采样
                if interpolation_mode == "bilinear":  # bilinear
                    output_data[n, h, w, :] = bilinear_interpolate(
                        input_data[n], x, y, h_in, w_in)
                elif interpolation_mode == "nearest":  # nearest
                    x_idx = int(x)
                    y_idx = int(y)
                    x_idx = np.clip(x_idx, 0, w_in - 1)
                    y_idx = np.clip(y_idx, 0, h_in - 1)
                    output_data[n, h, w, :] = input_data[n, y_idx, x_idx, :]
    return output_data


class GridSampleNet(nn.Cell):
    """Reshape and cache operation for NZ/ND format with all parameters"""

    @jit_for_graph_mode
    def construct(self, input_data, grid, interpolation_mode, padding_mode, align_corners):
        return ms_custom_ops.grid_sample(input_data, grid, interpolation_mode, padding_mode, align_corners)


def run_grid_sample(net, exec_mode, input_dtype, grid_dtype, align_corners, padding_mode, interpolation_mode, n, c, h_in, w_in, h_out, w_out, is_profiler=False):
    np_input = np.random.random((n, h_in, w_in, c)).astype(input_dtype)
    np_grid = np.random.uniform(-1, 1, (n, h_out, w_out, 2)).astype(grid_dtype)
    input_data = Tensor(np_input)
    grid = Tensor(np_grid)
    np_output = golden_grid_sample(np_input, np_grid, align_corners, padding_mode, interpolation_mode)
    if is_profiler == False:
        output_data = net(input_data, grid, interpolation_mode, padding_mode, align_corners)
        np.testing.assert_allclose(np_output, output_data.asnumpy(), rtol=1e-4, atol=1e-4, err_msg=" grid_sample ")
    else:
        profiler = Profiler(profiler_level=ProfilerLevel.Level2,
                    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                    aic_metrics=AicoreMetrics.AiCoreNone)
        for i in range(50):
            output_data = net(input_data, grid, interpolation_mode, padding_mode, align_corners)
        profiler.analyse()


@pytest.mark.level0 
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("input_dtype", [np.float32])
@pytest.mark.parametrize("grid_dtype", [np.float32])
@pytest.mark.parametrize("align_corners", [False])
@pytest.mark.parametrize("padding_mode", ["border"])
@pytest.mark.parametrize("interpolation_mode", ["bilinear"])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("c", [1536])
@pytest.mark.parametrize("h_in,w_in", [(24, 24)])
@pytest.mark.parametrize("h_out,w_out", [(1024, 1)])
def test_rope_v3_interleave(exec_mode, input_dtype, grid_dtype, align_corners, padding_mode, interpolation_mode, n, c, h_in, w_in, h_out, w_out):
    """
    Feature:aclnnGridSample kernel.
    Description: test for GridSampleExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net =  GridSampleNet()
    run_grid_sample(net, exec_mode, input_dtype, grid_dtype, align_corners, padding_mode, interpolation_mode, n, c, h_in, w_in, h_out, w_out)
