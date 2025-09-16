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
import sys
import numpy as np
import pytest
from functools import wraps
from mindspore import Profiler, Tensor, context, ops, mint, Parameter
import mindspore as ms
from mindspore.common.np_dtype import bfloat16
import mindspore.ops as ops
import mindspore.nn as nn
from mindspore import context, Tensor
from mindspore._c_expression import MSContext
import ms_custom_ops


def jit(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        if ms.get_context("mode") == "PYNATIVE_MODE":
            return func(*args, **kwargs)
        return ms.jit(func, jit_level="O0", infer_boost="on")(*args, **kwargs)

    return decorator


class AsdFusedAddTopKDivCustom(ms.nn.Cell):
    def __init__(self):
        super().__init__()

    @jit
    def construct(
        self, x, add_num, group_num, group_topk, n, k, activate_type, is_norm, scale
    ):
        return ms_custom_ops.fused_add_topk_div(
            x, add_num, group_num, group_topk, n, k, activate_type, is_norm, scale
        )


def compare(out, expect, dtype):
    if dtype == ms.float16:
        limit = 0.001
    elif dtype == ms.float32:
        limit = 0.0001
    elif dtype == ms.bfloat16:
        limit = 0.03
    else:
        raise ValueError("Unsupported dtype")

    out_flatten = out.flatten()
    expect_flatten = expect.flatten()

    err_cnt = 0
    size = len(out_flatten)
    err_cnt = np.sum(
        (np.abs(out_flatten - expect_flatten) / np.abs(expect_flatten) > limit).astype(
            np.int32
        )
    )
    limit_cnt = int(size * limit)
    if err_cnt > limit_cnt:
        print("[FAILED] err_cnt = ", err_cnt, "/", limit_cnt)
        return False
    else:
        print("[SUCCESS] err_cnt = ", err_cnt, "/", limit_cnt)
        return True


def numpy_topk(arr, k, axis=-1):
    # 获取排序后的元素索引
    arr_np = arr.asnumpy()
    sorted_indices = np.argsort(arr_np, axis=axis)
    # 根据排序方向获取前 k 个元素的索引
    if axis < 0:
        axis = arr_np.ndim + axis
    topk_indices = np.take(sorted_indices, np.arange(-k, 0), axis=axis)
    # 根据索引获取前 k 个元素
    topk_values = np.take_along_axis(arr_np, topk_indices, axis=axis)
    return topk_values, topk_indices


def golden_np(input, token_num, expert_num, group_num, k, k_inner):
    input0 = input.reshape((token_num, group_num, expert_num // group_num))
    output = np.copy(input0)
    input0 = input0.astype(np.float32)
    group_tensor, _ = numpy_topk(input0, k_inner)
    group_tensor = np.sum(group_tensor, axis=-1)
    # The torch version of the CI is too old. Not support the stable parameter in torch.argsort.
    sort_index = np.argsort(-group_tensor, kind="stable")
    cols_to_use = np.arange(k, group_num, dtype=np.int64)
    row_indices = np.repeat(np.arange(sort_index.shape[0]), cols_to_use.shape[0])
    col_indices = sort_index[:, cols_to_use].reshape(-1)
    output[row_indices, col_indices] = 0

    return np.reshape(output, (token_num, expert_num))


def before_fused(x_t, add_num_t, group_num, group_topk, n, scale):
    # golden (before fused)
    index_arr = Tensor(np.arange(1024, dtype=np.int32))
    index_arr_t = Tensor(np.array(index_arr, dtype=np.int32))
    x_t_32 = ops.cast(x_t, ms.float32)
    sigmoid_out = ops.sigmoid(x_t_32)
    add_out = sigmoid_out + add_num_t.astype(np.float32)
    # ops.auto_generate.group_topk精度不对
    a, b = add_out.shape
    group_topk_result = golden_np(add_out, a, b, group_num, group_topk, n)
    group_topk_tensor = ms.Tensor(group_topk_result)

    _, idx = mint.topk(group_topk_tensor, group_num)
    idx_32 = ops.cast(idx, ms.int32)
    gather_out = ops.gather(sigmoid_out, idx_32, 1, 1)

    sum_out = mint.sum(gather_out, -1, True)
    div_out = gather_out / sum_out
    mul_out = div_out * scale

    return mul_out, idx


def fused_add_topk_div(
    a,
    b,
    group_num,
    group_topk,
    n,
    k,
    mstype,
    mode,
    is_dyn=False,
    use_api=False,
    profile=False,
):
    os.environ["USE_LLM_CUSTOM_MATMUL"] = "off"
    os.environ["INTERNAL_PRINT_TILING"] = "on"
    os.environ["MS_DISABLE_INTERNAL_KERNELS_LIST"] = ""
    os.environ["MS_ENABLE_INTERNAL_BOOST"] = "off"
    context.set_context(mode=mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    # context.set_context(save_graphs=1, save_graphs_path="./fused_add_topk_div_graph")

    # 固定参数
    activate_type = 0  # 算子只支持0
    is_norm = True  # True时 会乘scale
    scale = 2.5  # 暂时固定

    x_np = np.random.randn(a, b)
    add_num_np = np.random.randn(b)
    x_t = Tensor(x_np).astype(mstype)
    add_num_t = Tensor(add_num_np).astype(mstype)
    # golden
    if profile:
        profiler = Profiler(start_profile=False, output_path="profiler")
        profiler.start()
        for i in range(50):
            golden_weight, golden_indices = before_fused(
                x_t, add_num_t, group_num, group_topk, n, scale
            )
        profiler.stop()
        profiler.analyse()
    golden_weight, golden_indices = before_fused(
        x_t, add_num_t, group_num, group_topk, n, scale
    )

    # expect
    net = AsdFusedAddTopKDivCustom()

    if use_api:
        if profile:
            profiler = Profiler(start_profile=False, output_path="profiler")
            profiler.start()
            for i in range(50):
                expect_weight, expect_indices = ms_custom_ops.fused_add_topk_div(
                    x_t,
                    add_num_t,
                    group_num,
                    group_topk,
                    n,
                    k,
                    activate_type,
                    is_norm,
                    scale,
                )
            profiler.stop()
            profiler.analyse()
            return
        expect_weight, expect_indices = ms_custom_ops.fused_add_topk_div(
            x_t, add_num_t, group_num, group_topk, n, k, activate_type, is_norm, scale
        )
    else:
        if profile:
            profiler = Profiler(start_profile=False, output_path="profiler")
            profiler.start()
            for i in range(50):
                expect_weight, expect_indices = net(
                    x_t,
                    add_num_t,
                    group_num,
                    group_topk,
                    n,
                    k,
                    activate_type,
                    is_norm,
                    scale,
                )
            profiler.stop()
            profiler.analyse()
            return
        if is_dyn:
            x_dyn = ms.Tensor(shape=[None, None], dtype=mstype)
            add_num_dyn = ms.Tensor(shape=[None], dtype=mstype)
            net.set_inputs(x=x_dyn, add_num=add_num_dyn)
            expect_weight, expect_indices = net(
                x_t,
                add_num_t,
                group_num,
                group_topk,
                n,
                k,
                activate_type,
                is_norm,
                scale,
            )
        else:
            expect_weight, expect_indices = net(
                x_t,
                add_num_t,
                group_num,
                group_topk,
                n,
                k,
                activate_type,
                is_norm,
                scale,
            )
    res = compare(expect_weight, golden_weight, mstype)
    assert res, "fused_add_topk_div compare failed."


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize(
    "input_and_params", [[8, 4, 2, 2, 2, 2], [1, 32, 8, 8, 2, 8], [2, 256, 8, 2, 2, 8]]
)
@pytest.mark.parametrize("use_api", [True, False])
@pytest.mark.parametrize("mstype", [ms.bfloat16, ms.float16, ms.float32])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.env_onecard
def test_asd_fused_add_topk_div_base(input_and_params, use_api, mstype, mode):
    """
    Feature: test asd_fused_add_topk_div operator in graph mode
    Description: test asd_fused_add_topk_div.
    Expectation: the result is correct
    """
    # group_topk <= group_num < expert
    # when b > 32, group_num must set to 8
    a, b, group_num, group_topk, n, k = input_and_params
    fused_add_topk_div(
        a=a,
        b=b,
        group_num=group_num,
        group_topk=group_topk,
        n=n,
        k=k,
        mstype=mstype,
        mode=mode,
        use_api=use_api,
        profile=False,
    )


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.parametrize("mstype", [ms.bfloat16, ms.float16, ms.float32])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.env_onecard
def test_asd_fused_add_topk_div_dynamic_shape(mstype, mode):
    """
    Feature: test asd_fused_add_topk_div operator in graph mode
    Description: test asd_fused_add_topk_div.
    # group_topk <= group_num < expert
    # when b > 32, group_num must set to 8"
    """
    a, b, group_num, group_topk, n, k = [8, 4, 2, 2, 2, 2]
    fused_add_topk_div(
        a=a,
        b=b,
        group_num=group_num,
        group_topk=group_topk,
        n=n,
        k=k,
        mstype=mstype,
        mode=mode,
        is_dyn=True,
    )


@pytest.mark.level0
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize(
    "input_and_params",
    [[1, 256, 8, 4, 2, 8], [11, 256, 8, 4, 2, 8], [8192, 256, 8, 4, 2, 8]],
)
@pytest.mark.parametrize("function_api", [False, True])
@pytest.mark.parametrize("is_dyn", [False, True])
@pytest.mark.parametrize("mstype", [ms.float16, ms.float32])
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.env_onecard
def test_internel_fused_add_topk_div_deepseek(
    input_and_params, function_api, is_dyn, mstype, mode
):
    """
    Feature: test asd_fused_add_topk_div operator in graph mode
    Description: test asd_fused_add_topk_div.
    Expectation: the result is correct
    """
    a, b, group_num, group_topk, n, k = input_and_params
    fused_add_topk_div(
        a=a,
        b=b,
        group_num=group_num,
        group_topk=group_topk,
        n=n,
        k=k,
        mstype=mstype,
        mode=mode,
        is_dyn=is_dyn,
        use_api=function_api,
        profile=False,
    )


if __name__ == "__main__":
    profiler = Profiler(start_profile=False, output_path="profiler")
    profiler.start()
    a = int(sys.argv[1])
    b = int(sys.argv[2])
    group_num = int(sys.argv[3])
    group_topk = int(sys.argv[4])
    n = int(sys.argv[5])
    k = int(sys.argv[6])
    fused_add_topk_div(
        a=a,
        b=b,
        group_num=group_num,
        group_topk=group_topk,
        n=n,
        k=k,
        mstype=ms.bfloat16,
        use_api=True,
        profile=False,
    )
    profiler.stop()
    profiler.analyse()
    exit()
