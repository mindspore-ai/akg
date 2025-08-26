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
import numpy as np
import pytest

import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore.common.np_dtype import bfloat16
import ms_custom_ops

class MoeGatingGroupTopkCell(nn.Cell):
    def __init__(self):
        super().__init__()

    def construct(self, x, bias=None, k=8, k_group=8, group_count=1, group_select_mode=0,
                  renorm=0, norm_type=0, out_flag=False, routed_scaling_factor=1.0, eps=1e-20):
        y, expert_idx, out = ms_custom_ops.moe_gating_group_topk(x, bias, k, k_group, group_count,
                                                        group_select_mode, renorm, norm_type,
                                                        out_flag, routed_scaling_factor, eps)
        return y, expert_idx, out


def get_ms_dtype(np_dtype):
    if np_dtype == np.float32:
        ms_dtype = ms.float32
    elif np_dtype == np.float16:
        ms_dtype = ms.float16
    elif np_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype


def np_softmax(x):
    max_x = np.max(x, axis=-1, keepdims=True)
    e_x = np.exp(x - max_x)
    sum_e_x = np.sum(e_x, axis=-1, keepdims=True)
    return e_x / sum_e_x


def np_max_dim(tensor, axis=-1, keepdims=False):
    """等效于 torch.max(input, dim, keepdim) -> 返回 (values, indices)"""
    values = np.max(tensor, axis=axis, keepdims=keepdims)
    indices = np.argmax(tensor, axis=axis)
    if keepdims:
        indices = np.expand_dims(indices, axis=axis)
    return values, indices


def np_arange(start=0, end=None, step=1, dtype=None, device='cpu'):
    """整合 torch.arange 的核心功能（忽略 device 参数）"""
    arr = np.arange(start, end, step, dtype)
    return arr


def sort_with_indices(x, index, descending=True, axis=-1):
    """
    对数组 x 排序，并同步调整 index 的顺序
    参数:
        x: np.ndarray, 待排序的数组
        index: np.ndarray 或 list, 与 x 同长度的索引序列
        descending: 是否降序排序 (默认升序)
        axis: 排序的轴向 (默认最后一个轴)
    返回:
        sorted_x: 排序后的 x
        sorted_index: 对应调整后的 index
    """
    # 检查维度一致性
    assert x.shape[axis] == index.shape[axis], "x 和 index 在排序轴上的长度必须一致"
    # 获取排序索引（支持降序）
    sort_order = -x if descending else x
    sorted_indices = np.argsort(sort_order, axis=axis, kind='stable')
    # 按排序索引调整 x 和 index
    sorted_x = np.take_along_axis(x, sorted_indices, axis=axis)
    sorted_index = np.take_along_axis(index, sorted_indices, axis=axis)
    return sorted_x, sorted_index


def np_golden(x_in, expert, group, k):
    x = x_in.astype(np.float32)
    scores = np_softmax(x)
    group_scores = scores.reshape(scores.shape[0], group, -1)
    topk_weights, topk_ids = np_max_dim(group_scores)
    group_ids = np_arange(0, expert, expert/group, np.int32)
    topk_ids = topk_ids + group_ids
    # topk_weights, topk_ids = sort_with_indices(topk_weights, topk_ids)
    return topk_weights, topk_ids, scores


def print_result(is_debug, out_flag, golden_softmax, y_softmax, golden_w, y, golden_idx, y_idx):
    if is_debug is not True:
        return
    if out_flag is True:
        print("\n==========softmax=========\n==golden==:\n{0}\n==kernel==:\n{1}".format(
            golden_softmax, y_softmax))
    print("\n==========score=========\n==golden==:\n{0}\n==kernel==:\n{1}".format(
        golden_w, y))
    print("\n==========index=========\n==golden==:\n{0}\n==kernel==:\n{1}".format(
        golden_idx, y_idx))
    print("\nkernel-score.max: {0}\nkernel-score.min: {1}\nkernel-idx.max: {2}\nkernel-idx.min: {3}\n".format(
        np.max(y.asnumpy()), np.min(y.asnumpy()), np.max(y_idx.asnumpy()), np.min(y_idx.asnumpy())))


def run(x_dtype, row, expert, k, k_group, group_count, group_select_mode, renorm,
        norm_type, out_flag, routed_scaling_factor, eps, is_dynamic, is_debug, run_mode=ms.GRAPH_MODE):
    ms.set_context(device_target="Ascend",
               mode=run_mode,
               jit_config={"jit_level": "O0", "infer_boost": "on"},
               pynative_synchronize=True,
                #   save_graphs=True,
                #   save_graphs_path="./moe_gating_group_topk_graph",
               )
    net = MoeGatingGroupTopkCell()
    if is_dynamic:
        x_shape = (row, expert)
        x_dynamic = ms.Tensor(
            shape=[None] * len(x_shape), dtype=get_ms_dtype(x_dtype))
        net.set_inputs(x_dynamic, None, k, k_group, group_count, group_select_mode,
                       renorm, norm_type, out_flag, routed_scaling_factor, eps)
        for item in range(1, 6):
            input_shape = (row + item, expert)
            x = np.random.uniform(-2, 2, input_shape).astype(x_dtype)
            x_tensor = ms.Tensor(x, dtype=get_ms_dtype(x_dtype))
            y, y_idx, y_softmax = net(x_tensor, None, k, k_group, group_count, group_select_mode,
                                      renorm, norm_type, out_flag, routed_scaling_factor, eps)
            golden_w, golden_idx, golden_softmax = np_golden(
                x, expert, group_count, k)
            np.testing.assert_allclose(golden_w, y.astype(ms.float32).asnumpy(),
                                       rtol=1e-2, atol=1e-2, err_msg='score 存在误差', verbose=True)
            np.testing.assert_allclose(golden_idx, y_idx.astype(ms.int32).asnumpy(),
                                       rtol=1e-2, atol=1e-2, err_msg='index 存在误差', verbose=True)
    else:
        x = np.random.uniform(-2, 2, (row, expert)).astype(x_dtype)
        golden_w, golden_idx, golden_softmax = np_golden(
            x, expert, group_count, k)
        x_tensor = ms.Tensor(x, dtype=get_ms_dtype(x_dtype))
        y, y_idx, y_softmax = net(x_tensor, None, k, k_group, group_count, group_select_mode,
                                  renorm, norm_type, out_flag, routed_scaling_factor, eps)
        print_result(is_debug, out_flag, golden_softmax,
                     y_softmax, golden_w, y, golden_idx, y_idx)
        np.testing.assert_allclose(golden_idx, y_idx.astype(ms.int32).asnumpy(),
                                   rtol=1e-2, atol=1e-2, err_msg='index error', verbose=True)
        np.testing.assert_allclose(golden_w, y.astype(ms.float32).asnumpy(),
                                   rtol=1e-2, atol=1e-2, err_msg='score error', verbose=True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('x_dtype', [np.float32, np.float16])
@pytest.mark.parametrize('row', [8000])
@pytest.mark.parametrize('expert', [64])
@pytest.mark.parametrize('k', [4])
@pytest.mark.parametrize('k_group', [4])
@pytest.mark.parametrize('group_count', [4])
@pytest.mark.parametrize('group_select_mode', [0])
@pytest.mark.parametrize('renorm', [0])
@pytest.mark.parametrize('norm_type', [0])
@pytest.mark.parametrize('out_flag', [False])
@pytest.mark.parametrize('routed_scaling_factor', [1.0])
@pytest.mark.parametrize('eps', [1e-20])
@pytest.mark.parametrize('is_dynamic', [True, False])
@pytest.mark.parametrize('is_debug', [False])
@pytest.mark.parametrize('run_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_moe_gating_group_topk_tp4(x_dtype, row, expert, k, k_group, group_count, group_select_mode, renorm,
                                   norm_type, out_flag, routed_scaling_factor, eps, is_dynamic, is_debug, run_mode):
    """
    Feature: 64专家，分4组，选出top4
    Description: What input in what scene
    Expectation: the result is correct
    """
    run(x_dtype, row, expert, k, k_group, group_count, group_select_mode, renorm,
        norm_type, out_flag, routed_scaling_factor, eps, is_dynamic, is_debug, run_mode)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize('x_dtype', [np.float32, np.float16])
@pytest.mark.parametrize('row', [8000])
@pytest.mark.parametrize('expert', [64])
@pytest.mark.parametrize('k', [8])
@pytest.mark.parametrize('k_group', [8])
@pytest.mark.parametrize('group_count', [8])
@pytest.mark.parametrize('group_select_mode', [0])
@pytest.mark.parametrize('renorm', [0])
@pytest.mark.parametrize('norm_type', [0])
@pytest.mark.parametrize('out_flag', [False])
@pytest.mark.parametrize('routed_scaling_factor', [1.0])
@pytest.mark.parametrize('eps', [1e-20])
@pytest.mark.parametrize('is_dynamic', [False, True])
@pytest.mark.parametrize('is_debug', [False])
@pytest.mark.parametrize('run_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_moe_gating_group_topk_tp8(x_dtype, row, expert, k, k_group, group_count, group_select_mode, renorm,
                                   norm_type, out_flag, routed_scaling_factor, eps, is_dynamic, is_debug, run_mode):
    """
    Feature: 64专家，分8组，选出top8
    Description: What input in what scene
    Expectation: the result is correct
    """
    run(x_dtype, row, expert, k, k_group, group_count, group_select_mode, renorm,
        norm_type, out_flag, routed_scaling_factor, eps, is_dynamic, is_debug, run_mode)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize('x_dtype', [bfloat16])
@pytest.mark.parametrize('row', [8000])
@pytest.mark.parametrize('expert', [64])
@pytest.mark.parametrize('k', [4])
@pytest.mark.parametrize('k_group', [4])
@pytest.mark.parametrize('group_count', [4])
@pytest.mark.parametrize('group_select_mode', [0])
@pytest.mark.parametrize('renorm', [0])
@pytest.mark.parametrize('norm_type', [0])
@pytest.mark.parametrize('out_flag', [False])
@pytest.mark.parametrize('routed_scaling_factor', [1.0])
@pytest.mark.parametrize('eps', [1e-20])
@pytest.mark.parametrize('is_dynamic', [True, False])
@pytest.mark.parametrize('is_debug', [False])
@pytest.mark.parametrize('run_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_moe_gating_group_topk_tp4_bf16(x_dtype, row, expert, k, k_group, group_count, group_select_mode, renorm,
                                   norm_type, out_flag, routed_scaling_factor, eps, is_dynamic, is_debug, run_mode):
    """
    Feature: 64专家，分4组，选出top4
    Description: What input in what scene
    Expectation: the result is correct
    """
    run(x_dtype, row, expert, k, k_group, group_count, group_select_mode, renorm,
        norm_type, out_flag, routed_scaling_factor, eps, is_dynamic, is_debug, run_mode)
