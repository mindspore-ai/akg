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
"""Tests for ms_custom_ops.ring_mla using numpy golden reference."""

from typing import List, Optional, Tuple
import math

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context, ops, nn
from mindspore.common.np_dtype import bfloat16 as np_bfloat16

import ms_custom_ops


class TestConfig:
    def __init__(self, device_target: str = "Ascend", mode: int = context.GRAPH_MODE):
        self.device_target = device_target
        self.mode = mode

    def apply(self):
        context.set_context(device_target=self.device_target, mode=self.mode)


def _make_triu_mask(size: int, dtype: np.dtype, batch: Optional[int] = None) -> np.ndarray:
    # Follow coefficient semantics similar to provided torch test code
    if dtype == np.float16:
        # mask values directly used
        base = -10000.0
        mask = np.triu(np.ones((size, size), dtype=np.float32) * base, 1)
    else:
        # bf16 and others: use a very negative number
        base = 1
        mask = np.triu(np.ones((size, size), dtype=np.float32), 1) * base
    if batch is not None:
        mask = np.broadcast_to(mask, (batch, size, size)).copy()
    return mask.astype(np.float32)


def _reconstruct_full(q_base: np.ndarray, q_rope: np.ndarray) -> np.ndarray:
    # q_base: [q_ntokens, heads, d_base], q_rope: [q_ntokens, heads, d_rope]
    return np.concatenate([q_base, q_rope], axis=-1)


def _expand_kv_to_heads(k_or_v: np.ndarray, heads: int, kv_heads: int) -> np.ndarray:
    # k_or_v: [kv_ntokens, kv_heads, dim]
    if heads == kv_heads:
        return k_or_v
    group_num = heads // kv_heads
    # Repeat along kv_head dim to match total heads
    return np.repeat(k_or_v, repeats=group_num, axis=1)


def _golden_attention(
    q_base: np.ndarray,
    q_rope: np.ndarray,
    k_base: np.ndarray,
    k_rope: np.ndarray,
    v: np.ndarray,
    mask: Optional[np.ndarray],
    q_seq_lens: List[int],
    kv_seq_lens: List[int],
    heads: int,
    kv_heads: int,
    scale: float,
    out_dim: int,
    out_dtype: np.dtype,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute golden attention and lse without ring update.

    Returns:
        out: [q_ntokens, heads, out_dim] in out_dtype
        lse: [heads, q_ntokens] in float32
    """
    q = _reconstruct_full(q_base, q_rope)  # [q_ntokens, heads, d]
    k = _reconstruct_full(k_base, k_rope)  # [kv_ntokens, kv_heads, d]
    v_dim = v.shape[-1]
    assert out_dim == v_dim

    # Expand K/V from kv_heads to heads by repeating per-group
    k_exp = _expand_kv_to_heads(k, heads, kv_heads)  # [kv_ntokens, heads, d]
    v_exp = _expand_kv_to_heads(v, heads, kv_heads)  # [kv_ntokens, heads, out_dim]

    q_ntokens = q.shape[0]
    kv_ntokens = k.shape[0]
    assert sum(q_seq_lens) == q_ntokens
    assert sum(kv_seq_lens) == kv_ntokens

    # Offsets per batch
    out = np.zeros((q_ntokens, heads, out_dim), dtype=np.float32)
    lse = np.zeros((heads, q_ntokens), dtype=np.float32)

    q_offset = 0
    kv_offset = 0
    batch = len(q_seq_lens)
    for b in range(batch):
        q_len = q_seq_lens[b]
        kv_len = kv_seq_lens[b]

        if q_len == 0:
            continue

        q_slice = q[q_offset : q_offset + q_len]  # [q_len, heads, d]
        if kv_len == 0:
            # When kv_len=0, define output as zeros and LSE as zeros to match op behavior
            out[q_offset : q_offset + q_len] = 0.0
            lse[:, q_offset : q_offset + q_len] = 0.0
            q_offset += q_len
            continue

        k_slice = k_exp[kv_offset : kv_offset + kv_len]  # [kv_len, heads, d]
        v_slice = v_exp[kv_offset : kv_offset + kv_len]  # [kv_len, heads, out_dim]

        # Compute per-head attention
        # logits[i, h, j] = dot(q_slice[i,h,:], k_slice[j,h,:]) * scale
        # We'll compute as batch matmul per head using einsum
        # q_slice: [q_len, heads, d], k_slice: [kv_len, heads, d]
        logits = np.einsum("qhd,khd->qhk", q_slice.astype(np.float32), k_slice.astype(np.float32)) * scale

        # Apply mask if provided
        if mask is not None:
            if mask.ndim == 2:
                mask_slice = mask[:q_len, :kv_len]
            elif mask.ndim == 3:
                mask_slice = mask[b, :q_len, :kv_len]
            elif mask.ndim == 4:
                # [batch, heads, q, kv]
                mask_slice = mask[b, :, :q_len, :kv_len]  # [heads, q, kv]
                # transpose to [q, heads, kv]
                mask_slice = np.transpose(mask_slice, (1, 0, 2))
            else:
                raise ValueError("Unsupported mask ndim")
            if mask.ndim < 4:
                # broadcast to [q, heads, kv] by expanding head axis
                mask_slice = np.broadcast_to(mask_slice[:, None, :], logits.shape).copy()
            logits = logits + mask_slice.astype(np.float32)

        # Softmax per head and query across kv axis
        m = np.max(logits, axis=2, keepdims=True)
        exp_logits = np.exp((logits - m).astype(np.float32))
        denom = np.sum(exp_logits, axis=2, keepdims=True)
        p = exp_logits / np.maximum(denom, 1e-38)

        # Output: [q_len, heads, out_dim]
        o = np.einsum("qhk,khd->qhd", p.astype(np.float32), v_slice.astype(np.float32))

        # LSE: [heads, q_len]
        lse_b = (np.log(np.maximum(denom.squeeze(-1), 1e-38)) + m.squeeze(-1)).transpose(1, 0)

        out[q_offset : q_offset + q_len] = o
        lse[:, q_offset : q_offset + q_len] = lse_b

        q_offset += q_len
        kv_offset += kv_len

    return out.astype(out_dtype), lse.astype(np.float32)


def _golden_ring_update(
    out_cur: np.ndarray,  # [q_ntokens, heads, out_dim]
    lse_cur: np.ndarray,  # [heads, q_ntokens]
    o_prev: np.ndarray,   # [q_ntokens, heads, out_dim]
    lse_prev: np.ndarray, # [heads, q_ntokens]
    q_seq_lens: List[int],
    kv_seq_lens: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    # Ring update with special handling for kv_len=0 cases
    # When kv_len=0 for a batch, keep the previous output unchanged

    batch = len(q_seq_lens)
    result_out = o_prev.copy().astype(np.float32)
    result_lse = lse_prev.copy().astype(np.float32)

    q_offset = 0
    for b in range(batch):
        q_len = q_seq_lens[b]
        kv_len = kv_seq_lens[b]

        if q_len == 0:
            continue

        if kv_len == 0:
            # When kv_len=0, keep previous output unchanged and set LSE to zeros
            result_lse[:, q_offset:q_offset + q_len] = 0.0
            q_offset += q_len
            continue

        # Normal ring update for this batch
        exp_new = np.exp(lse_cur[:, q_offset:q_offset + q_len].astype(np.float32))
        exp_old = np.exp(lse_prev[:, q_offset:q_offset + q_len].astype(np.float32))

        # Align shapes
        exp_new_e = np.transpose(exp_new, (1, 0))[:, :, None]  # [q_len, heads, 1]
        exp_old_e = np.transpose(exp_old, (1, 0))[:, :, None]  # [q_len, heads, 1]

        num = (out_cur[q_offset:q_offset + q_len].astype(np.float32) * exp_new_e +
               o_prev[q_offset:q_offset + q_len].astype(np.float32) * exp_old_e)
        den = exp_new_e + exp_old_e

        result_out[q_offset:q_offset + q_len] = num / np.maximum(den, 1e-38)
        result_lse[:, q_offset:q_offset + q_len] = np.log(np.maximum(exp_new + exp_old, 1e-38))

        q_offset += q_len

    return result_out, result_lse


def _ms_tensor(x: np.ndarray) -> Tensor:
    if x.dtype == np_bfloat16:
        # MindSpore expects float32 array then cast by dtype
        return Tensor(x.astype(np.float32)).astype(ms.bfloat16)
    return Tensor(x)


def _compare_output_data(out: np.ndarray, golden: np.ndarray, np_dtype: np.dtype,
                        heads: int, max_seq: int) -> bool:
    """
    Advanced precision comparison based on data type and computation complexity.

    Args:
        out: Output data from operator
        golden: Golden reference data
        np_dtype: Data type (np.float16, np_bfloat16, or np.float32)
        heads: Number of attention heads
        max_seq: Maximum sequence length

    Returns:
        bool: True if precision test passes
    """
    import logging

    # Flatten tensors for element-wise comparison
    golden_flat = golden.flatten().astype(np.float32)
    out_flat = out.flatten().astype(np.float32)
    out_len = out_flat.shape[0]

    # Calculate absolute differences
    diff = np.abs(golden_flat - out_flat)
    max_diff = np.max(diff)

    # Legacy standard with fixed ratios
    if np_dtype == np_bfloat16:
        ratios = [0.001, 0.001, 0.005, 0.005]  # [rel_loose, abs_loose, rel_strict, abs_strict]
    else:  # fp16
        ratios = [0.001, 0.001, 0.005, 0.005]

    limit_error = np.maximum(np.abs(golden_flat) * ratios[0], ratios[1])
    strict_limit_error = np.maximum(np.abs(golden_flat) * ratios[2], ratios[3])
    error_count = np.sum(diff > limit_error)
    strict_error_count = np.sum(diff > strict_limit_error)

    accuracy_loose = 1.0 - float(error_count) / out_len
    accuracy_strict = 1.0 - float(strict_error_count) / out_len

    logging.info(f"Max difference: {max_diff:.6f}")
    logging.info(f"Loose accuracy (1/1000): {accuracy_loose:.6f}")
    logging.info(f"Strict accuracy (5/1000): {accuracy_strict:.6f}")

    # New standard: adaptive threshold based on data type and complexity
    calc_times = heads * max_seq + 4

    if np_dtype == np_bfloat16:
        if calc_times < 2048:
            error_factor = 2**(-7)  # ~0.0078
        else:
            error_factor = 2**(-6)  # ~0.0156
    elif np_dtype == np.float16:
        if calc_times < 2048:
            error_factor = 2**(-8)  # ~0.0039
        else:
            error_factor = 2**(-7)  # ~0.0078
    else:  # float32
        if calc_times < 2048:
            error_factor = 2**(-11)  # ~0.00049
        elif calc_times < 16384:
            error_factor = 2**(-10)  # ~0.00098
        else:
            error_factor = 2**(-14)  # ~0.000061

    # Adaptive threshold: max(|golden|, 1.0) * error_factor
    error_threshold = np.maximum(np.abs(golden_flat), 1.0) * error_factor
    adaptive_pass = np.all(diff <= error_threshold)

    logging.info(f"Calculation complexity: {calc_times}")
    logging.info(f"Error factor: {error_factor:.6e}")
    logging.info(f"Adaptive precision test: {'PASS' if adaptive_pass else 'FAIL'}")

    # Legacy fallback check
    if np_dtype == np_bfloat16:
        legacy_pass = (float(strict_error_count) / out_len) <= ratios[2]
    else:
        legacy_pass = (float(strict_error_count) / out_len) <= ratios[0]

    logging.info(f"Legacy precision test: {'PASS' if legacy_pass else 'FAIL'}")

    # Return True if either test passes (more robust)
    return adaptive_pass or legacy_pass


def _init_prev_tensors(rng: np.random.Generator, q_ntokens: int, heads: int, dv: int,
                       dtype: np.dtype, is_ring: int) -> Tuple[np.ndarray, np.ndarray]:
    if is_ring == 1:
        o_prev = rng.uniform(-1.0, 1.0, size=(q_ntokens, heads, dv)).astype(dtype)
        lse_prev = (rng.random((heads, q_ntokens)) * 10.0).astype(np.float32)
    else:
        o_prev = np.zeros((q_ntokens, heads, dv), dtype=dtype)
        lse_prev = np.zeros((heads, q_ntokens), dtype=np.float32)
    return o_prev, lse_prev


class RingMLANet(nn.Cell):
    """Thin wrapper to call ms_custom_ops.ring_mla with fixed attributes."""

    def __init__(self, head_num: int, scale_value: float, kv_head_num: int, mask_type: int, calc_type: int):
        super().__init__()
        self.head_num = head_num
        self.scale_value = scale_value
        self.kv_head_num = kv_head_num
        self.mask_type = mask_type
        self.calc_type = calc_type
        # determine execution mode once during initialization
        self._is_pynative = (context.get_context("mode") == context.PYNATIVE_MODE)

    def construct(self, q_nope, q_rope, key, k_rope, value, mask, alibi_coeff,
                  deq_scale_qk, deq_offset_qk, deq_scale_pv, deq_offset_pv, quant_p, log_n, o_prev, lse_prev,
                  q_seq_lens, context_lens):
        if self._is_pynative:
            q_lens_cpu = q_seq_lens.move_to("CPU")
            kv_lens_cpu = context_lens.move_to("CPU")
        else:
            q_lens_cpu = ops.move_to(q_seq_lens, "CPU")
            kv_lens_cpu = ops.move_to(context_lens, "CPU")
        return ms_custom_ops.ring_mla(
            q_nope, q_rope, key, k_rope, value, mask, alibi_coeff,
            deq_scale_qk, deq_offset_qk, deq_scale_pv, deq_offset_pv, quant_p, log_n, o_prev, lse_prev,
            q_lens_cpu, kv_lens_cpu,
            self.head_num, self.scale_value, self.kv_head_num, self.mask_type, self.calc_type)


class RingMLATestCase:
    """A comprehensive test case for ring multi-head latent attention (MLA) operations.
    
    This class encapsulates all the necessary components for testing ring MLA functionality,
    including input generation, mask creation, golden reference computation, and comparison
    with MindSpore implementation. It supports various configurations such as different
    data types (fp16, bf16), mask types (none, triu), and sequence lengths for both
    queries and key-values.
    """

    def __init__(
        self,
        *,
        heads: int,
        kv_heads: int,
        dim_qk: int,
        dim_v: int,
        q_seq_lens: List[int],
        kv_seq_lens: List[int],
        np_dtype: np.dtype,
        mask_type: int,  # 0: no mask, 1: triu
        is_ring: int,
        rng_seed: int,
        mask_size: Optional[int] = None,
    ):
        self.heads = heads
        self.kv_heads = kv_heads
        self.dim_qk = dim_qk
        self.dim_v = dim_v
        self.q_seq_lens = q_seq_lens
        self.kv_seq_lens = kv_seq_lens
        self.np_dtype = np_dtype
        self.mask_type = mask_type
        self.is_ring = is_ring
        self.rng = np.random.default_rng(rng_seed)
        self.q_ntokens = int(sum(q_seq_lens))
        self.kv_ntokens = int(sum(kv_seq_lens))
        self.d_base = 128
        self.d_rope = dim_qk - self.d_base
        self.scale = 1.0 / math.sqrt(float(dim_qk))
        self.max_seq = max(max(q_seq_lens), max(kv_seq_lens))
        self.mask_size = mask_size if mask_size is not None else self.max_seq

    def build_inputs(self):
        q_full = self.rng.uniform(-1.0, 1.0, size=(self.q_ntokens, self.heads, self.dim_qk)).astype(self.np_dtype)
        k_full = self.rng.uniform(-1.0, 1.0, size=(self.kv_ntokens, self.kv_heads, self.dim_qk)).astype(self.np_dtype)
        v = self.rng.uniform(-1.0, 1.0, size=(self.kv_ntokens, self.kv_heads, self.dim_v)).astype(self.np_dtype)
        q_base, q_rope = q_full[..., : self.d_base], q_full[..., self.d_base :]
        k_base, k_rope = k_full[..., : self.d_base], k_full[..., self.d_base :]
        return q_base, q_rope, k_base, k_rope, v

    def build_masks(self, batch: Optional[int] = None):
        if self.mask_type == 0:
            return None, None
        assert self.mask_size == 512
        # fp16: both op and golden use the same values
        if self.np_dtype == np.float16:
            mask = _make_triu_mask(self.mask_size, np.float16, batch)
            return mask.astype(np.float16), mask.astype(np.float32)
        # bf16: op uses structural bf16 mask, golden uses -3e38 fp32
        base = np.triu(np.ones((self.mask_size, self.mask_size), dtype=np.float32), 1)
        if batch is not None:
            base = np.broadcast_to(base, (batch, self.mask_size, self.mask_size)).copy()
        mask_op = base.astype(np_bfloat16)
        mask_golden = base * -3e38
        return mask_op, mask_golden

    def run(self, run_mode: int, dynamic: bool = False):
        q_base, q_rope, k_base, k_rope, v = self.build_inputs()
        assert len(self.q_seq_lens) == len(self.kv_seq_lens)
        batch = len(self.q_seq_lens)
        mask_op, mask_golden = self.build_masks(batch=batch)

        # Golden
        out_dtype = np.float16 if self.np_dtype == np.float16 else np_bfloat16
        cur_out, cur_lse = _golden_attention(
            q_base, q_rope, k_base, k_rope, v,
            mask_golden if mask_golden is not None else None,
            self.q_seq_lens, self.kv_seq_lens,
            self.heads, self.kv_heads, self.scale, self.dim_v, out_dtype,
        )
        o_prev, lse_prev = _init_prev_tensors(self.rng, self.q_ntokens, self.heads, self.dim_v, self.np_dtype, is_ring=self.is_ring)
        if self.is_ring == 1:
            golden_out, golden_lse = _golden_ring_update(cur_out.astype(np.float32), cur_lse, o_prev.astype(np.float32), lse_prev, self.q_seq_lens, self.kv_seq_lens)
        else:
            golden_out, golden_lse = cur_out, cur_lse

        # Net
        calc_type = 0 if self.is_ring == 1 else 1
        net = RingMLANet(self.heads, self.scale, self.kv_heads, self.mask_type, calc_type)

        # Optionally enable dynamic shape by setting input placeholders
        if dynamic:
            ms_dtype = ms.float16 if self.np_dtype == np.float16 else ms.bfloat16
            # query no rope / rope
            q_nope_dyn = Tensor(shape=[None, self.heads, self.d_base], dtype=ms_dtype)
            q_rope_dyn = Tensor(shape=[None, self.heads, self.d_rope], dtype=ms_dtype)
            # key / rope / value
            k_nope_dyn = Tensor(shape=[None, self.kv_heads, self.d_base], dtype=ms_dtype)
            k_rope_dyn = Tensor(shape=[None, self.kv_heads, self.d_rope], dtype=ms_dtype)
            v_dyn = Tensor(shape=[None, self.kv_heads, self.dim_v], dtype=ms_dtype)
            # mask (optional)
            if self.mask_type == 0:
                mask_dyn = None
            else:
                mask_dtype = ms.float16 if self.np_dtype == np.float16 else ms.bfloat16
                mask_dyn = Tensor(shape=[None, self.mask_size, self.mask_size], dtype=mask_dtype)
            # optional tensors left as None
            alibi_dyn = None
            deq_scale_qk_dyn = None
            deq_offset_qk_dyn = None
            deq_scale_pv_dyn = None
            deq_offset_pv_dyn = None
            quant_p_dyn = None
            log_n_dyn = None
            # previous outputs and lse
            o_prev_dyn = Tensor(shape=[None, self.heads, self.dim_v], dtype=ms_dtype)
            lse_prev_dyn = Tensor(shape=[self.heads, None], dtype=ms.float32)
            # sequence length tensors
            q_lens_dyn = Tensor(shape=[None], dtype=ms.int32)
            kv_lens_dyn = Tensor(shape=[None], dtype=ms.int32)

            net.set_inputs(
                q_nope_dyn, q_rope_dyn,
                k_nope_dyn, k_rope_dyn,
                v_dyn, mask_dyn,
                alibi_dyn, deq_scale_qk_dyn, deq_offset_qk_dyn, deq_scale_pv_dyn, deq_offset_pv_dyn, quant_p_dyn, log_n_dyn,
                o_prev_dyn, lse_prev_dyn,
                q_lens_dyn, kv_lens_dyn,
            )
        out, lse = net(
            _ms_tensor(q_base), _ms_tensor(q_rope),
            _ms_tensor(k_base), _ms_tensor(k_rope),
            _ms_tensor(v), _ms_tensor(mask_op) if mask_op is not None else None,
            None, None, None, None, None, None, None,
            _ms_tensor(o_prev), _ms_tensor(lse_prev),
            _ms_tensor(np.array(self.q_seq_lens, dtype=np.int32)),
            _ms_tensor(np.array(self.kv_seq_lens, dtype=np.int32)),
        )

        # Compare using advanced precision validation
        out_np = (out.float().asnumpy() if self.np_dtype == np_bfloat16 else out.asnumpy()).astype(np.float32)
        lse_np = lse.asnumpy().astype(np.float32)

        # Test output precision
        out_pass = _compare_output_data(
            out_np, golden_out.astype(np.float32),
            self.np_dtype, self.heads, self.max_seq
        )

        # Test LSE precision with simpler validation
        lse_pass = _compare_output_data(
            lse_np, golden_lse.astype(np.float32),
            self.np_dtype, self.heads, self.max_seq
        )

        assert out_pass, "Output precision test failed"
        assert lse_pass, "LSE precision test failed"

@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('is_ring', [0, 1])
@pytest.mark.parametrize('dynamic', [True, False])
def test_ring_mla_fp16_no_mask(run_mode, is_ring, dynamic):
    cfg = TestConfig(device_target="Ascend", mode=run_mode)
    cfg.apply()
    case = RingMLATestCase(
        heads=16, kv_heads=16, dim_qk=192, dim_v=128,
        q_seq_lens=[100, 100], kv_seq_lens=[100, 100], np_dtype=np.float16,
        mask_type=0, is_ring=is_ring, rng_seed=2025 + is_ring,
    )
    case.run(run_mode, dynamic=dynamic)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('is_ring', [0, 1])
@pytest.mark.parametrize('dynamic', [True, False])
def test_ring_mla_fp16_mask(run_mode, is_ring, dynamic):
    cfg = TestConfig(device_target="Ascend", mode=run_mode)
    cfg.apply()
    case = RingMLATestCase(
        heads=16, kv_heads=16, dim_qk=192, dim_v=128,
        q_seq_lens=[150, 50], kv_seq_lens=[200, 200], np_dtype=np.float16,
        mask_type=1, is_ring=is_ring, rng_seed=2026 + is_ring, mask_size=512,
    )
    case.run(run_mode, dynamic=dynamic)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('is_ring', [0, 1])
@pytest.mark.parametrize('dynamic', [True, False])
def test_ring_mla_bf16_no_mask(run_mode, is_ring, dynamic):
    cfg = TestConfig(device_target="Ascend", mode=run_mode)
    cfg.apply()
    case = RingMLATestCase(
        heads=16, kv_heads=16, dim_qk=192, dim_v=128,
        q_seq_lens=[128, 128], kv_seq_lens=[128, 128], np_dtype=np_bfloat16,
        mask_type=0, is_ring=is_ring, rng_seed=2027 + is_ring,
    )
    case.run(run_mode, dynamic=dynamic)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('is_ring', [0, 1])
@pytest.mark.parametrize('dynamic', [True, False])
def test_ring_mla_bf16_mask(run_mode, is_ring, dynamic):
    cfg = TestConfig(device_target="Ascend", mode=run_mode)
    cfg.apply()
    case = RingMLATestCase(
        heads=16, kv_heads=16, dim_qk=192, dim_v=128,
        q_seq_lens=[120, 72], kv_seq_lens=[192, 192], np_dtype=np_bfloat16,
        mask_type=1, is_ring=is_ring, rng_seed=2028 + is_ring, mask_size=512,
    )
    case.run(run_mode, dynamic=dynamic)


@pytest.mark.level0
@pytest.mark.platform_ascend910b
@pytest.mark.env_onecard
@pytest.mark.parametrize('run_mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('is_ring', [0, 1])
@pytest.mark.parametrize('dynamic', [True, False])
def test_ring_mla_bf16_mask_diff_qkv_lens(run_mode, is_ring, dynamic):
    cfg = TestConfig(device_target="Ascend", mode=run_mode)
    cfg.apply()
    case = RingMLATestCase(
        heads=16, kv_heads=16, dim_qk=192, dim_v=128,
        q_seq_lens=[64, 128, 32, 1, 100], kv_seq_lens=[200, 180, 50, 10, 128], np_dtype=np_bfloat16,
        mask_type=1, is_ring=is_ring, rng_seed=2029 + is_ring, mask_size=512,
    )
    case.run(run_mode, dynamic=dynamic)

