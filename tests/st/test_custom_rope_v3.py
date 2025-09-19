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

def golden_apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = mint.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return mint.cat((o1, o2), dim=-1)
    else:
        return mint.stack((o1, o2), dim=-1).flatten(-2)

def golden_apply_rotary_emb_split(net, exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim, rotary_dim, is_profiler=False):
    cos_head_dim = rotary_dim // 2
    np_query = np.random.random((tokens, head_num_q, head_dim))
    np_key = np.random.random((tokens, head_num_k, head_dim))
    np_cos = np.random.random((tokens, cos_head_dim))
    np_sin = np.random.random((tokens, cos_head_dim))
    query = Tensor(np_query, dtype=query_dtype)
    key = Tensor(np_key , dtype=query_dtype)
    cos = Tensor(np_cos, dtype=query_dtype)
    sin = Tensor(np_sin, dtype=query_dtype)
    golden_q = golden_apply_rotary_emb(query, cos, sin, False)
    golden_k = golden_apply_rotary_emb(key, cos, sin, False)
    if is_profiler == False:
        out_query, out_key = net(query, key, cos, sin, layout, rotary_mode)
        np.testing.assert_allclose(golden_q.asnumpy(), out_query.asnumpy(), rtol=1e-4, atol=1e-4, err_msg=" query ")
        np.testing.assert_allclose(golden_k.asnumpy(), out_key.asnumpy(), rtol=1e-4, atol=1e-4, err_msg=" key ")
    else:
        profiler = Profiler(profiler_level=ProfilerLevel.Level2,
                    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                    aic_metrics=AicoreMetrics.AiCoreNone)
        for i in range(50):
            out_query, out_key = net(query, key, cos, sin, layout, rotary_mode)
        profiler.analyse()

class ApplyRotaryEmbV3Net(nn.Cell):
    """Reshape and cache operation for NZ/ND format with all parameters"""
    
    @jit_for_graph_mode
    def construct(self, query, key, cos, sin, layout, rotary_mode):
        return ms_custom_ops.apply_rotary_pos_emb_v3(query, key, cos, sin, layout, rotary_mode)

def run_rope_interleave(net, exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim, is_profiler=False):
    cos_head_dim = head_dim // 2
    np_query = np.random.random((tokens, head_num_q, head_dim))
    np_key = np.random.random((tokens, head_num_k, head_dim))
    np_cos = np.random.random((tokens, cos_head_dim))
    np_sin = np.random.random((tokens, cos_head_dim))
    query = Tensor(np_query, dtype=query_dtype)
    key = Tensor(np_key , dtype=query_dtype)
    cos = Tensor(np_cos, dtype=query_dtype)
    sin = Tensor(np_sin, dtype=query_dtype)
    golden_q = golden_apply_rotary_emb(query, cos, sin, False)
    golden_k = golden_apply_rotary_emb(key, cos, sin, False)
    if is_profiler == False:
        out_query, out_key = net(query, key, cos, sin, layout, rotary_mode)
        np.testing.assert_allclose(golden_q.asnumpy(), out_query.asnumpy(), rtol=1e-4, atol=1e-4, err_msg=" query ")
        np.testing.assert_allclose(golden_k.asnumpy(), out_key.asnumpy(), rtol=1e-4, atol=1e-4, err_msg=" key ")
    else:
        profiler = Profiler(profiler_level=ProfilerLevel.Level2,
                    activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
                    aic_metrics=AicoreMetrics.AiCoreNone)
        for i in range(50):
            out_query, out_key = net(query, key, cos, sin, layout, rotary_mode)
        profiler.analyse()


def run_rope_interleave_split(net, exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim, rotary_dim, is_profiler=False):
    cos_head_dim = rotary_dim // 2
    np_query = np.random.random((tokens, head_num_q, head_dim))
    np_key = np.random.random((tokens, head_num_k, head_dim))
    np_cos = np.random.random((tokens, cos_head_dim))
    np_sin = np.random.random((tokens, cos_head_dim))
    query = Tensor(np_query, dtype=query_dtype)
    key = Tensor(np_key , dtype=query_dtype)
    cos = Tensor(np_cos, dtype=query_dtype)
    sin = Tensor(np_sin, dtype=query_dtype)
    if is_profiler == False:    
        query_rot = query[..., :rotary_dim]
        query_pass = query[..., rotary_dim:]

        key_rot = key[..., :rotary_dim]
        key_pass = key[..., rotary_dim:]

        query_rot = golden_apply_rotary_emb(query_rot, cos, sin, False)
        key_rot = golden_apply_rotary_emb(key_rot, cos, sin, False)
        
        golden_q = mint.cat((query_rot, query_pass), dim=-1)
        golden_k = mint.cat((key_rot, key_pass), dim=-1)
    else:
        start_time = time.perf_counter()
        for i in range(50):
            query_rot = query[..., :rotary_dim]
            query_pass = query[..., rotary_dim:]

            key_rot = key[..., :rotary_dim]
            key_pass = key[..., rotary_dim:]

            query_rot = golden_apply_rotary_emb(query_rot, cos, sin, False)
            key_rot = golden_apply_rotary_emb(key_rot, cos, sin, False)
            
            query = mint.cat((query_rot, query_pass), dim=-1)
            key = mint.cat((key_rot, key_pass), dim=-1)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("tokens:", tokens)
        print("小算子50次平均耗时(毫秒)：", total_time * 1000/50)


    if is_profiler == False:
        out_query, out_key = net(query, key, cos, sin, layout, rotary_mode)
        np.testing.assert_allclose(golden_q.asnumpy(), out_query.asnumpy(), rtol=1e-4, atol=1e-4, err_msg=" query ")
        np.testing.assert_allclose(golden_k.asnumpy(), out_key.asnumpy(), rtol=1e-4, atol=1e-4, err_msg=" key ")
    else:
        # profiler = Profiler(profiler_level=ProfilerLevel.Level2,
        #             activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
        #             aic_metrics=AicoreMetrics.AiCoreNone)
        # for i in range(50):
        #     out_query, out_key = net(query, key, cos, sin, layout, rotary_mode)
        # profiler.analyse()

        start_time = time.perf_counter()
        for i in range(50):
            out_query, out_key = net(query, key, cos, sin, layout, rotary_mode)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print("tokens:", tokens)
        print("大算子50次平均耗时(毫秒)：", total_time * 1000/50)

@pytest.mark.level0 
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("query_dtype", [ms.float32, ms.float16])
@pytest.mark.parametrize("layout", ["BSH"])
@pytest.mark.parametrize("rotary_mode", ["interleave"])
@pytest.mark.parametrize("tokens", [10, 40960])
@pytest.mark.parametrize("head_num_q", [32])
@pytest.mark.parametrize("head_num_k", [2])
@pytest.mark.parametrize("head_dim", [64])
def test_rope_v3_interleave(exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim):
    """
    Feature:aclnnApplyRotaryPosEmb kernel.
    Description: test for ApplyRotaryPosEmbExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net =  ApplyRotaryEmbV3Net()
    run_rope_interleave(net, exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim)

@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("query_dtype", [ms.float32, ms.float16])
@pytest.mark.parametrize("layout", ["BSH"])
@pytest.mark.parametrize("rotary_mode", ["interleave"])
@pytest.mark.parametrize("tokens", [10, 40960])
@pytest.mark.parametrize("head_num_q", [32])
@pytest.mark.parametrize("head_num_k", [2])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("rotary_dim", [64])
def test_rope_v3_interleave_split(exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim, rotary_dim):
    """
    Feature:aclnnApplyRotaryPosEmb kernel.
    Description: test for ApplyRotaryPosEmbExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net =  ApplyRotaryEmbV3Net()
    run_rope_interleave_split(net, exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim, rotary_dim)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE])
@pytest.mark.parametrize("query_dtype", [ms.float16])
@pytest.mark.parametrize("layout", ["BSH"])
@pytest.mark.parametrize("rotary_mode", ["interleave"])
@pytest.mark.parametrize("tokens", [2048])
@pytest.mark.parametrize("head_num_q", [16])
@pytest.mark.parametrize("head_num_k", [2])
@pytest.mark.parametrize("head_dim", [64])
def test_rope_v3_interleave_profiler(exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim):
    """
    Feature:aclnnApplyRotaryPosEmb kernel.
    Description: test for ApplyRotaryPosEmbExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net =  ApplyRotaryEmbV3Net()
    run_rope_interleave(net, exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim, True)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE])
@pytest.mark.parametrize("query_dtype", [ms.float16])
@pytest.mark.parametrize("layout", ["BSH"])
@pytest.mark.parametrize("rotary_mode", ["interleave"])
@pytest.mark.parametrize("tokens", [20])
@pytest.mark.parametrize("head_num_q", [32])
@pytest.mark.parametrize("head_num_k", [2])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("rotary_dim", [64])
def test_rope_v3_interleave_split_profil(exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim, rotary_dim):
    """
    Feature:aclnnApplyRotaryPosEmb kernel.
    Description: test for ApplyRotaryPosEmbExt ops.
    Expectation:should pass for all testcases.
    """
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net =  ApplyRotaryEmbV3Net()
    run_rope_interleave_split(net, exec_mode, query_dtype, layout, rotary_mode, tokens, head_num_q, head_num_k, head_dim, rotary_dim, True)
