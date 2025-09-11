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
import numpy as np
import pytest
from functools import wraps

import mindspore.ops as ops
import mindspore.nn as nn
import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.np_dtype import bfloat16
from mindspore._c_expression import MSContext
import ms_custom_ops


def get_ms_dtype(query_dtype):
    if query_dtype == np.float32:
        ms_dtype = ms.float32
    elif query_dtype == np.float16:
        ms_dtype = ms.float16
    elif query_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype


def apply_rotary_pos_emb_ext(query, key, cos, sin, layout, rotary_mode="half"):
    """
    自定义算子：应用旋转位置编码V2版本
    参数:
        query: 输入query张量，4维，形状为[batch, seq_len, num_heads, head_dim]
        key: 输入key张量，4维，形状为[batch, seq_len, num_heads, head_dim]
        cos: 余弦位置编码，4维，形状为[batch, seq_len, 1, head_dim]
        sin: 正弦位置编码，4维，形状为[batch, seq_len, 1, head_dim]
        layout: 布局格式，目前只支持1(BSND)
        rotary_mode: 旋转模式，支持"half", "quarter", "interleave"
    返回:
        q_embed: 旋转位置编码后的query
        k_embed: 旋转位置编码后的key
    """
    if rotary_mode == "half":
        return apply_rotary_pos_emb_half(query, key, cos, sin)
    elif rotary_mode == "quarter":
        return apply_rotary_pos_emb_quarter(query, key, cos, sin)
    elif rotary_mode == "interleave":
        return apply_rotary_pos_emb_interleave(query, key, cos, sin)
    else:
        raise ValueError(f"Unsupported rotary mode: {rotary_mode}")


def apply_rotary_pos_emb_half(query, key, cos, sin):
    """Half模式旋转位置编码的numpy实现(Golden函数)"""
    # 处理query
    query_q1 = query[..., : query.shape[-1] // 2]
    query_q2 = query[..., query.shape[-1] // 2 :]
    query_rotate = np.concatenate((-query_q2, query_q1), axis=-1)
    q_embed = query * cos + query_rotate * sin

    # 处理key
    key_k1 = key[..., : key.shape[-1] // 2]
    key_k2 = key[..., key.shape[-1] // 2 :]
    key_rotate = np.concatenate((-key_k2, key_k1), axis=-1)
    k_embed = key * cos + key_rotate * sin

    return q_embed, k_embed


def apply_rotary_pos_emb_quarter(query, key, cos, sin):
    """Quarter模式旋转位置编码的numpy实现(Golden函数)"""
    # 处理query
    quarter_idx = query.shape[-1] // 4
    half_idx = query.shape[-1] // 2
    three_quarter_idx = query.shape[-1] // 4 * 3

    query_q1 = query[..., :quarter_idx]
    query_q2 = query[..., quarter_idx:half_idx]
    query_q3 = query[..., half_idx:three_quarter_idx]
    query_q4 = query[..., three_quarter_idx:]

    query_rotate = np.concatenate((-query_q2, query_q1, -query_q4, query_q3), axis=-1)
    q_embed = query * cos + query_rotate * sin

    # 处理key
    key_q1 = key[..., :quarter_idx]
    key_q2 = key[..., quarter_idx:half_idx]
    key_q3 = key[..., half_idx:three_quarter_idx]
    key_q4 = key[..., three_quarter_idx:]

    key_rotate = np.concatenate((-key_q2, key_q1, -key_q4, key_q3), axis=-1)
    k_embed = key * cos + key_rotate * sin

    return q_embed, k_embed


def apply_rotary_pos_emb_interleave(query, key, cos, sin):
    """Interleave模式旋转位置编码的numpy实现(Golden函数)"""
    # 处理query
    query_q1 = query[..., ::2]
    query_q2 = query[..., 1::2]

    # 重塑形状以便拼接
    orig_shape = query.shape
    query_q1_flat = query_q1.reshape(-1, 1)
    query_q2_flat = query_q2.reshape(-1, 1)

    query_rotate_flat = np.concatenate((-query_q2_flat, query_q1_flat), axis=-1)
    query_rotate = query_rotate_flat.reshape(orig_shape)

    q_embed = query * cos + query_rotate * sin

    # 处理key
    key_q1 = key[..., ::2]
    key_q2 = key[..., 1::2]

    key_q1_flat = key_q1.reshape(-1, 1)
    key_q2_flat = key_q2.reshape(-1, 1)

    key_rotate_flat = np.concatenate((-key_q2_flat, key_q1_flat), axis=-1)
    key_rotate = key_rotate_flat.reshape(key.shape)

    k_embed = key * cos + key_rotate * sin

    return q_embed, k_embed


def jit(func):
    @wraps(func)
    def decorator(*args, **kwargs):
        if ms.get_context("mode") == "PYNATIVE_MODE":
            return func(*args, **kwargs)
        return ms.jit(func, jit_level="O0", infer_boost="on")(*args, **kwargs)

    return decorator


class ApplyRotaryPosEmbNet(ms.nn.Cell):
    def _init__(self):
        super().__init__()

    @jit
    def construct(self, query, key, cos, sin, layout, rotary_mode):
        query_embed, key_embed = ms_custom_ops.apply_rotary_pos_emb_ext(
            query, key, cos, sin, layout, rotary_mode
        )
        return query_embed, key_embed


def run(
    net,
    base,
    cos_dtype,
    seq_len,
    batch_size,
    num_head,
    hidden_dim,
    max_seq_len,
    query_dtype,
    pos_dtype,
    ndim,
    cos_format,
    rotary_mode="half",
):
    query_data = np.random.uniform(
        0, 1, [batch_size, seq_len, num_head, hidden_dim]
    ).astype(query_dtype)
    key_data = np.random.uniform(
        0, 1, [batch_size, seq_len, num_head, hidden_dim]
    ).astype(query_dtype)
    cos_data = np.random.uniform(0, 1, [batch_size, seq_len, 1, hidden_dim]).astype(
        query_dtype
    )
    sin_data = cos_data = np.random.uniform(
        0, 1, [batch_size, seq_len, 1, hidden_dim]
    ).astype(query_dtype)

    query1 = query_data
    query2 = query1.copy()

    key1 = key_data
    key2 = key1.copy()

    cos1 = cos_data
    cos2 = cos1.copy()
    sin1 = sin_data
    sin2 = sin1.copy()

    golden_query_emb, golden_key_emb = apply_rotary_pos_emb_ext(
        query1, key1, cos1, sin1, "BSND", rotary_mode
    )

    query2 = Tensor(query2, dtype=get_ms_dtype(query_dtype))
    key2 = Tensor(key2, dtype=get_ms_dtype(query_dtype))
    cos2 = Tensor(cos2, dtype=get_ms_dtype(query_dtype))
    sin2 = Tensor(sin2, dtype=get_ms_dtype(query_dtype))

    custom_query_emb, custom_key_emb = net(
        query2, key2, cos2, sin2, "BSND", rotary_mode
    )
    np.testing.assert_allclose(golden_query_emb, custom_query_emb, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(golden_key_emb, custom_key_emb, rtol=1e-2, atol=1e-2)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.parametrize("exec_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("query_dtype", [np.float16])
@pytest.mark.parametrize("cos_dtype", [np.float16])
@pytest.mark.parametrize("cos_format", [2])
@pytest.mark.parametrize("rotary_mode", ["half"])
@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [1, 256, 512, 1024])
@pytest.mark.parametrize("num_head", [16, 32])
def test_rope_float16(
    exec_mode,
    query_dtype,
    cos_dtype,
    cos_format,
    rotary_mode,
    batch_size,
    seq_len,
    num_head,
):
    """
    Feature:aclnnApplyRotaryPosEmb kernel.
    Description: test for ApplyRotaryPosEmbExt ops.
    Expectation:should pass for all testcases.
    """
    ndim = 4
    hidden_dim = 128
    base = 10000
    max_seq_len = seq_len
    ms.set_context(device_target="Ascend", mode=exec_mode)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    net = ApplyRotaryPosEmbNet()
    run(
        net,
        base,
        cos_dtype,
        seq_len,
        batch_size,
        num_head,
        hidden_dim,
        max_seq_len,
        query_dtype,
        np.int32,
        ndim,
        cos_format,
        rotary_mode,
    )
