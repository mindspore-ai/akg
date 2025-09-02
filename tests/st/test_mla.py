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

"""test mla"""

import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.ops.operations.nn_ops import PagedAttention
import numpy as np
import pytest
import ms_custom_ops


class MlaTestParam:
    """MlaTestParam"""

    def __init__(self, num_heads, kv_heads, block_size, head_size_nope, head_size_rope, num_blocks,
                 q_seq_lens: list, context_lengths: list, tor, nope_ms_dtype, rope_ms_dtype, mask_type: str,
                 is_quant_flag=False, run_mode=ms.GRAPH_MODE):

        self.num_heads = num_heads
        self.kv_heads = kv_heads
        self.block_size = block_size
        self.head_size_nope = head_size_nope
        self.head_size_rope = head_size_rope
        self.num_blocks = num_blocks
        self.q_seq_lens = q_seq_lens
        self.context_lengths = context_lengths
        self.tor = tor
        self.is_quant_flag = is_quant_flag
        self.nope_ms_dtype = nope_ms_dtype
        self.rope_ms_dtype = rope_ms_dtype
        self.mask_type = mask_type
        self.mask_factor = -10000.0 if rope_ms_dtype == ms.float16 else 1.0

        self.batch = len(q_seq_lens)

        self.max_context_len = max(context_lengths)
        self.max_num_blocks_per_seq = (
            self.max_context_len + block_size - 1) // block_size

        self.num_tokens = (int)(np.array(q_seq_lens).sum())
        self.block_tables = self._build_block_tables()

        self._build_tensor_inputs()

        self.run_mode = run_mode

    def _build_np_mask(self):
        """_build_np_mask"""
        pre_qseqlen = 0
        np_ori_pa_mask = np.zeros(shape=(self.num_tokens, self.max_context_len)).astype(np.float32)
        for i in range(self.batch):
            qseqlen = self.q_seq_lens[i]
            kseqlen = self.context_lengths[i]
            tri = np.ones((qseqlen, qseqlen))
            tri = np.triu(tri, 1)
            tri *= self.mask_factor
            np_ori_pa_mask[pre_qseqlen:(pre_qseqlen + qseqlen), kseqlen-qseqlen:kseqlen] = tri
            pre_qseqlen += qseqlen
        self.ori_pa_mask_tensor = Tensor(np_ori_pa_mask, dtype=self.rope_ms_dtype)

        if self.mask_type == "MASK_NONE":
            return None


        if self.mask_type == "MASK_SPEC":
            pre_qseqlen = 0
            np_mask = np.zeros(
                shape=(self.num_tokens, self.max_context_len)).astype(np.float32)
            for i in range(self.batch):
                qseqlen = self.q_seq_lens[i]
                kseqlen = self.context_lengths[i]
                tri = np.ones((qseqlen, qseqlen))
                tri = np.triu(tri, 1)
                tri *= -10000.0
                np_mask[pre_qseqlen:(pre_qseqlen + qseqlen),
                        kseqlen-qseqlen:kseqlen] = tri
                pre_qseqlen += qseqlen
            return np_mask

        if self.mask_type == "MASK_FREE":
            # [[-10000.0 -10000.0 -10000.0 ... -10000.0],
            # [0        -10000.0 -10000.0 ... -10000.0],
            # [0               0 -10000.0 ... -10000.0],
            # ...
            # [0               0        0 ... -10000.0],
            # [0               0        0 ...        0]]
            q_len = max(self.q_seq_lens)
            mask_free = np.full((125 + 2 * q_len, 128), -10000.0)
            mask_free = np.triu(mask_free, 2 - q_len)
            return mask_free

        return None


    def _build_block_tables(self):
        """_build_block_tables"""
        block_tables_list = []
        for i in range(self.num_tokens):
            block_table = [
                i * self.max_num_blocks_per_seq + _ for _ in range(self.max_num_blocks_per_seq)
            ]
            block_tables_list.append(block_table)

        return block_tables_list


    def _build_tensor_inputs(self):
        """_build_tensor_inputs"""
        np_q_nope = np.random.uniform(-1.0, 1.0, size=(
            self.num_tokens, self.num_heads, self.head_size_nope))
        np_q_rope = np.random.uniform(-1.0, 1.0, size=(
            self.num_tokens, self.num_heads, self.head_size_rope))
        np_ctkv = np.random.uniform(-1.0, 1.0, size=(self.num_blocks, self.block_size,
                                                     self.kv_heads, self.head_size_nope))
        np_k_rope = np.random.uniform(-1.0, 1.0, size=(self.num_blocks, self.block_size,
                                                       self.kv_heads, self.head_size_rope))

        np_context_lens = np.array(self.context_lengths).astype(np.int32)
        np_q_seq_lens = np.array(self.q_seq_lens).astype(np.int32)

        self.q_nope_tensor = Tensor(np_q_nope, dtype=self.nope_ms_dtype)
        self.q_rope_tensor = Tensor(np_q_rope, dtype=self.rope_ms_dtype)
        self.ctkv_tensor = ms.Parameter(Tensor(np_ctkv, dtype=self.nope_ms_dtype), name="ctkv")
        self.k_rope_tensor = ms.Parameter(Tensor(np_k_rope, dtype=self.rope_ms_dtype), name="k_rope")

        self.block_tables_tensor = Tensor(
            np.array(self.block_tables).astype(np.int32))

        np_mask = self._build_np_mask()
        self.mask_tensor = None if np_mask is None else Tensor(
            np_mask, dtype=self.rope_ms_dtype)

        if self.nope_ms_dtype == ms.int8:
            self.deq_scale_qk_tensor = Tensor(
                np.random.uniform(-1.0, 1.0, size=(self.num_heads,)), dtype=ms.float32)
            self.deq_scale_pv_tensor = Tensor(
                np.random.uniform(-1.0, 1.0, size=(self.num_heads,)), dtype=ms.float32)
        else:
            self.deq_scale_qk_tensor = None
            self.deq_scale_pv_tensor = None

        self.q_seq_lens_tensor = Tensor(np_q_seq_lens)
        self.context_lengths_tensor = Tensor(np_context_lens)


class Net(nn.Cell):
    """Net"""

    def __init__(self, q_head_num, kv_head_num, mask_type, tor):
        super().__init__()
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.mask_type = mask_type
        self.tor = tor
        self._ispynative = (context.get_context("mode") == context.PYNATIVE_MODE)

    def construct(self, q_nope, q_rope, ctkv, k_rope, block_tables, mask, deq_scale_qk, deq_scale_pv,
                  q_seq_lens, batch_valid_length, input_format=0):
        if self._ispynative:
            q_lens_cpu = q_seq_lens.move_to("CPU")
            kv_lens_cpu = batch_valid_length.move_to("CPU")
        else:
            q_lens_cpu = ops.move_to(q_seq_lens, "CPU")
            kv_lens_cpu = ops.move_to(batch_valid_length, "CPU")

        return ms_custom_ops.mla(q_nope, q_rope, ctkv, k_rope, block_tables, mask, deq_scale_qk,
                                     deq_scale_pv, q_lens_cpu, kv_lens_cpu, self.q_head_num, self.tor,
                                     self.kv_head_num, self.mask_type, input_format=input_format)


class GoldenNet(nn.Cell):
    """GoldenNet"""

    def __init__(self, q_head_num, kv_head_num, mask_type, tor, mla_v_dim):
        super().__init__()
        self.q_head_num = q_head_num
        self.kv_head_num = kv_head_num
        self.mask_type = mask_type
        self.tor = tor
        self.mla_v_dim = mla_v_dim
        self.op = PagedAttention(self.q_head_num, self.tor, self.kv_head_num, 'DEFAULT', 'MASK_DEFAULT',
                                 self.mla_v_dim)

    def construct(self, query, key_cache, value_cache, block_tables, batch_valid_length, antiquant_scale,
                  antiquant_offset, attn_mask, q_seq_lens, alibi_mask):
        return self.op(query, key_cache, value_cache, block_tables, batch_valid_length, antiquant_scale,
                       antiquant_offset, attn_mask, q_seq_lens, alibi_mask)


def run_mla(test_param: MlaTestParam):
    """run mla"""
    context.set_context(mode=test_param.run_mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    dyn_q_nope_shape = [None for _ in test_param.q_nope_tensor.shape]
    dyn_q_nope_tensor = Tensor(
        shape=dyn_q_nope_shape, dtype=test_param.q_nope_tensor.dtype)

    if test_param.mask_type == "MASK_NONE":
        mask_type = 0
    elif test_param.mask_type == "MASK_SPEC":
        mask_type = 3
    elif test_param.mask_type == "MASK_FREE":
        mask_type = 4
    else:
        mask_type = -1

    net = Net(test_param.num_heads, test_param.kv_heads,
              mask_type, test_param.tor)
    net.set_inputs(q_nope=dyn_q_nope_tensor)
    net.phase = "increment"

    ctkv_tensor = test_param.ctkv_tensor
    k_rope_tensor = test_param.k_rope_tensor
    input_format = 0
    if test_param.is_quant_flag:
        ctkv_tensor = ms.jit(ms_custom_ops.trans_data)(ctkv_tensor, 1)
        k_rope_tensor = ms.jit(ms_custom_ops.trans_data)(k_rope_tensor, 1)
        input_format = 1

    out, _ = net(test_param.q_nope_tensor, test_param.q_rope_tensor, ctkv_tensor, k_rope_tensor,
                 test_param.block_tables_tensor, test_param.mask_tensor, test_param.deq_scale_qk_tensor,
                 test_param.deq_scale_pv_tensor, test_param.q_seq_lens_tensor, test_param.context_lengths_tensor,
                 input_format)
    return out


def run_golden(test_param: MlaTestParam):
    """run_golden"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    mla_v_dim = 512
    query = ops.reshape(ops.concat((test_param.q_nope_tensor, test_param.q_rope_tensor), axis=-1),
                        (test_param.num_tokens, 1, -1))
    key_cache = ops.concat(
        (test_param.ctkv_tensor, test_param.k_rope_tensor), axis=-1)
    dyn_q_shape = [None for _ in test_param.q_nope_tensor.shape]
    dyn_q_nope_tensor = Tensor(
        shape=dyn_q_shape, dtype=test_param.q_nope_tensor.dtype)
    golden_net = GoldenNet(test_param.num_heads, test_param.kv_heads,
                           "MASK_DEFAULT", test_param.tor, mla_v_dim)
    golden_net.set_inputs(query=dyn_q_nope_tensor)

    out_golden = golden_net(query, key_cache, key_cache, test_param.block_tables_tensor,
                            test_param.context_lengths_tensor, None, None, test_param.ori_pa_mask_tensor,
                            test_param.q_seq_lens_tensor, None)

    return out_golden


class GoldenNumpy:
    """GoldenNumpy"""
    def __init__(self, max_context_length, num_heads, block_size, head_size_rope, head_size_nope, is_quant_flag=False,
                 deq_scale_qk=None, deq_scale_pv=None):
        self.is_quant_flag = is_quant_flag
        self.deq_scale_qk = deq_scale_qk
        self.deq_scale_pv = deq_scale_pv
        self.block_size = block_size
        self.num_heads = num_heads
        self.kvsplit = 1
        self.max_context_len = max_context_length


    def softmax_quant_inner(self, x, is_first):
        """softmax_quant_inner"""
        x_max = np.max(x, axis=-1, keepdims=True)
        if is_first:
            g_max = x_max
            self.dm = 0
        else:
            g_max = np.maximum(self.global_max, x_max)
            self.dm = self.global_max - g_max
        self.global_max = g_max
        exp = np.exp(x - g_max)
        row_sum = np.sum(exp, axis=-1, keepdims=True)
        row_maxp = np.max(exp, axis=-1, keepdims=True)
        scale = row_maxp.astype("float32") / 127.0
        int8_res = exp / scale
        res = int8_res.astype("float16")
        res = np.rint(res).astype("int8")
        deq_scale_v_new = self.deq_scale_pv * row_maxp[:, 0, 0] / 127
        return res, row_sum, deq_scale_v_new, g_max, self.dm


    def group_mm(self, heads, group_num, A, B, deq_scale):
        """group_mm"""
        group_head = heads // group_num
        score_fp32 = None
        for i in range(group_num):
            if self.is_quant_flag:
                group_score_int32 = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.int32),
                                              B[i: (i+1), :, :].astype(np.int32)).astype(np.int32)
                group_score_fp32 = group_score_int32.astype(np.float32) *\
                    deq_scale[(i * group_head): (i + 1) * group_head].reshape(group_head, 1, 1).astype(np.float32)
            else:
                group_score_fp32 = np.matmul(A[i * group_head: (i + 1) * group_head, :, :].astype(np.float32),
                                             B[i:(i + 1), :, :].astype(np.float32))
            if score_fp32 is None:
                score_fp32 = group_score_fp32
            else:
                score_fp32 = np.concat((score_fp32, group_score_fp32), 0)
        return score_fp32


    def softmax_quant(self, x, heads, kv_head, value):
        """softmax_quant"""
        # (kv_heads, context_len, head_size)
        kv_seqlen = value.shape[1]
        cur_kv_seqlen = kv_seqlen
        n_loop = (cur_kv_seqlen + self.block_size - 1) // self.block_size
        qk_n = self.block_size
        self.tmp_l_list = []
        self.tmp_o_list = []
        for cur_idx in range(self.kvsplit):
            kv_seqlen_align = (kv_seqlen + self.block_size - 1) // self.block_size  * self.block_size
            start_kv = cur_idx * self.max_context_len
            cur_kv_seqlen = self.max_context_len
            kv_loop = (kv_seqlen_align + self.max_context_len - 1) // self.max_context_len
            if cur_idx >= kv_loop:
                continue
            if cur_idx == (kv_loop - 1):
                cur_kv_seqlen = kv_seqlen - cur_idx * self.max_context_len
            n_loop = (cur_kv_seqlen + self.block_size - 1) // self.block_size
            qk_n = self.block_size
            end_kv = start_kv
            for n_idx in range(n_loop):
                is_first_iter = (n_idx == 0)
                if n_idx == n_loop - 1:
                    qk_n = cur_kv_seqlen - n_idx * self.block_size
                end_kv = end_kv + qk_n
                block = x[:, :, start_kv : end_kv]
                p_block, l_l, deq_scale_v_new, _, dm = self.softmax_quant_inner(block, is_first_iter)
                self.deq_scale_v_new = deq_scale_v_new
                value_block = value[:, start_kv : end_kv, :]
                l_o = self.group_mm(heads, kv_head, p_block, value_block, self.deq_scale_v_new)
                if n_idx == 0:
                    self.g_l = l_l
                    self.g_o = l_o
                else:
                    dm = np.exp(dm)
                    self.g_l = self.g_l * dm
                    self.g_l = self.g_l + l_l
                    self.g_o = self.g_o * dm
                    self.g_o = self.g_o + l_o
                start_kv = start_kv + qk_n
            self.g_o = self.g_o / self.g_l
            self.tmp_o_list.append(self.g_o.reshape([1, self.num_heads, 1, value.shape[2]]))
            ls = np.log(self.g_l) + self.global_max
            self.tmp_l_list.append(ls.reshape([1, self.num_heads]))
        if self.kvsplit > 1:
            l = np.concat(self.tmp_l_list, 0)
            o = np.concat(self.tmp_o_list, 0)
            l = np.transpose(l, (1, 0))
            lse_max = np.max(l, axis=1, keepdims=True)
            lse_sum = np.sum(np.exp(l - lse_max), axis=1, keepdims=True)
            lse_log_sum = np.log(lse_sum) + lse_max
            scale = np.exp(l - lse_log_sum)
            o = o * scale.transpose(1, 0)[:, :, np.newaxis, np.newaxis]
            self.g_o = np.sum(o, axis=0, keepdims=True)
            self.g_o = np.squeeze(self.g_o, axis=0)
        return self.g_o


    def softmax_float(self, x):
        """softmax_float"""
        row_max = np.max(x, axis=-1, keepdims=True)
        exp = np.exp(x - row_max)
        row_sum = np.sum(exp, axis=-1, keepdims=True)
        res = exp / row_sum
        return res


    def single_attention(self, q_nope, key, value, tor: float, data_type, query_rope, key_rope, mask=None):
        """single_attention"""
        # Q * K.T
        q_nope = np.transpose(q_nope, (1, 0, 2))
        if self.is_quant_flag:
            query_rope = np.transpose(query_rope, (1, 0, 2))
            key_rope = np.transpose(key_rope, (1, 2, 0))

        key = np.transpose(key, (1, 2, 0))
        qk_res = self.group_mm(q_nope.shape[0], key.shape[0], q_nope, key, self.deq_scale_qk)  # (head_num, q_seqlen, k_seqlen)
        if self.is_quant_flag:
            self.is_quant_flag = False
            qk_rope_res = self.group_mm(query_rope.shape[0], key_rope.shape[0], query_rope, key_rope, None)
            self.is_quant_flag = True
            qk_res = qk_res + qk_rope_res
        qk_res = qk_res.astype(np.float32) * tor

        if mask is not None:
            qk_res = qk_res + mask

        if self.is_quant_flag:
            self.global_max = np.full([q_nope.shape[0], 1, 1], np.finfo(np.float32).min)
            p_high, _, deq_scale_v_new, _, _ = self.softmax_quant_inner(qk_res, 1)
            self.deq_scale_v_new = deq_scale_v_new
            value = np.transpose(value, (1, 0, 2))
            s_qk = qk_res
            out = self.softmax_quant(s_qk, q_nope.shape[0], key.shape[0], value)
        else:
            # softmax
            p_high = self.softmax_float(qk_res)
            p = p_high.astype(data_type)

            # P * V
            value = np.transpose(value, (1, 0, 2))
            out = self.group_mm(q_nope.shape[0], key.shape[0], p, value, None)
            out = np.transpose(out, (1, 0, 2))
        return out


    def do_mla_numpy(self, output, q_nope, ctkv, block_tables, q_seq_lens, context_lens, mask,
                     tor, data_type, query_rope_input, key_rope_input):
        """do_mla_numpy"""
        num_heads = q_nope.shape[1]
        kv_heads = ctkv.shape[2]
        head_size_nope = ctkv.shape[3]
        block_size = ctkv.shape[1]

        index = 0
        q_rope = None
        batch = len(q_seq_lens)
        is_mtp = int(max(q_seq_lens) > 1)

        for i in range(batch):
            block_table = block_tables[i]
            context_len = int(context_lens[i])
            q_seq_len = int(q_seq_lens[i])
            if context_len == 0:
                continue

            q = q_nope[index:index + q_seq_len].reshape(q_seq_len, num_heads, head_size_nope)
            if self.is_quant_flag:
                q_rope = query_rope_input[index:index + q_seq_len].reshape(q_seq_len, num_heads, 64)
            keys = []
            values = []
            key_ropes = []
            for j in range(context_len):
                block_number = int(block_table[j // block_size])
                block_offset = j % block_size

                k = ctkv[block_number, block_offset, :, :]
                k = k.reshape(kv_heads, head_size_nope)
                keys.append(k)
                if self.is_quant_flag:
                    k_rope = key_rope_input[block_number, block_offset, :, :]
                    k_rope = k_rope.reshape(kv_heads, 64)
                    key_ropes.append(k_rope)

                v = ctkv[block_number, block_offset, :, :]
                v = v.reshape(kv_heads, head_size_nope)
                values.append(v)
            keys = np.stack(keys, axis=0)
            if self.is_quant_flag:
                key_ropes = np.stack(key_ropes, axis=0)
            values = np.stack(values, axis=0)
            local_mask = mask[index:index + q_seq_len, :context_len] if is_mtp else None
            out = self.single_attention(q, keys, values, tor, data_type, q_rope, key_ropes, local_mask)
            out = out.reshape(q_seq_len, num_heads, head_size_nope)
            output[index:index + q_seq_len] = out.astype(data_type)
            index = index + q_seq_len


def run_golden_numpy(test_param: MlaTestParam):
    """run_golden_numpy"""
    shape_out = (test_param.num_tokens, test_param.num_heads, test_param.head_size_nope)

    nope_np_dtype = np.float16 if test_param.nope_ms_dtype == ms.float16 else np.float32
    output = np.zeros(shape_out, dtype=nope_np_dtype)

    max_context_length = max(test_param.context_lengths_tensor.asnumpy())
    deq_scale_qk = test_param.deq_scale_qk_tensor.asnumpy() if test_param.deq_scale_qk_tensor is not None else None
    deq_scale_pv = test_param.deq_scale_pv_tensor.asnumpy() if test_param.deq_scale_pv_tensor is not None else None
    golden = GoldenNumpy(max_context_length, test_param.num_heads, test_param.block_size, test_param.head_size_rope,
                         test_param.head_size_nope, test_param.is_quant_flag,
                         deq_scale_qk, deq_scale_pv)

    is_mtp = int(max(test_param.q_seq_lens) > 1)
    if is_mtp:
        numpy_mask_factor = 1.0 if nope_np_dtype == np.float16 else -10000.0
        mask = test_param.ori_pa_mask_tensor.asnumpy().astype(np.float32) * numpy_mask_factor
    else:
        mask = None
    golden.do_mla_numpy(output, test_param.q_nope_tensor.asnumpy(),
                        test_param.ctkv_tensor.asnumpy(),
                        test_param.block_tables_tensor.asnumpy(),
                        test_param.q_seq_lens_tensor.asnumpy(),
                        test_param.context_lengths_tensor.asnumpy(),
                        mask,
                        test_param.tor, nope_np_dtype,
                        test_param.q_rope_tensor.asnumpy(), test_param.k_rope_tensor.asnumpy())

    return output


def run_test(test_param: MlaTestParam):
    """run test"""
    out_golden = run_golden(test_param)
    out_actual = run_mla(test_param)

    assert np.allclose(out_actual.astype(ms.float32).asnumpy().reshape(-1),
                       out_golden.astype(ms.float32).asnumpy().reshape(-1), 0.001, 0.001)


def run_test_with_numpy_golden(test_param: MlaTestParam):
    """run test"""
    out_actual = run_mla(test_param)
    out_golden = run_golden_numpy(test_param)

    assert np.allclose(out_actual.astype(ms.float32).asnumpy().reshape(-1),
                       out_golden.astype(np.float32).reshape(-1), 0.001, 0.001)


# block_num = 8 batch = 128 failed
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('batch', [4, 128])
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mla_base(dtype, batch, mode):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    q_seq_lens = [1] * batch
    context_lengths = [np.random.randint(192, 200) for _ in range(batch)] #[192, 193, 194, 195]
    test_param = MlaTestParam(32, 1, 128, 512, 64, 1024, q_seq_lens,
                              context_lengths, 0.001, dtype, dtype, "MASK_NONE", run_mode=mode)
    run_test(test_param)


# int8 need set MS_INTERNAL_ENABLE_NZ_OPS="Mla"
@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mask_type', ["MASK_NONE"])
@pytest.mark.parametrize("q_seq_lens", [[1, 1, 1, 1]])
@pytest.mark.parametrize('dtype', [ms.bfloat16, ms.float16])
@pytest.mark.parametrize('q_head_num', [32, 96])
@pytest.mark.parametrize('block_size', [16, 128])
def test_mla_int8(mask_type, q_seq_lens, dtype, q_head_num, block_size):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    context_lengths = [192, 193, 194, 195]
    test_param = MlaTestParam(q_head_num, 1, block_size, 512, 64, 1024, q_seq_lens, context_lengths, 0.001,
                              ms.int8, dtype, mask_type, True)
    run_test_with_numpy_golden(test_param)


# int8 does not support mtp
@pytest.mark.skip
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mask_type', ["MASK_SPEC"])
@pytest.mark.parametrize("q_seq_lens", [[1, 1, 3, 1]])
@pytest.mark.parametrize('dtype', [ms.bfloat16])
@pytest.mark.parametrize('q_head_num', [32])
@pytest.mark.parametrize('block_size', [128])
def test_mla_int8_mtp(mask_type, q_seq_lens, dtype, q_head_num, block_size):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    context_lengths = [192, 193, 194, 195]
    test_param = MlaTestParam(q_head_num, 1, block_size, 512, 64, 1024, q_seq_lens, context_lengths, 0.001,
                              ms.int8, dtype, mask_type, True)
    run_test_with_numpy_golden(test_param)


# 'block_size', [16, 32, 64], 'q_head_num', [128] failed
# when q_head_num = 128, block_size must be 128
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('block_size', [16, 32, 64, 128])
@pytest.mark.parametrize('q_head_num', [16, 32, 64])
def test_mla_block_size_q_head_num(block_size, q_head_num):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    q_seq_lens = [1, 1, 1, 1]
    context_lengths = [192, 193, 194, 195]
    test_param = MlaTestParam(q_head_num, 1, block_size, 512, 64, 1024, q_seq_lens, context_lengths,
                              0.001, ms.float16, ms.float16, "MASK_NONE")
    run_test(test_param)


# 'q_head_num', [128] 'block_size' [64] failed
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
@pytest.mark.parametrize('block_size', [64, 128])
@pytest.mark.parametrize('q_head_num', [64, 32])
def test_mla_mtp_mask_spec(dtype, block_size, q_head_num):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    q_seq_lens = [1, 4, 2, 1]
    context_lengths = [192, 193, 194, 195]
    test_param = MlaTestParam(q_head_num, 1, block_size, 512, 64, 128, q_seq_lens, context_lengths,
                              0.001, dtype, dtype, "MASK_SPEC")
    run_test(test_param)


# q_head_num = 128, 'block_size', [32, 64] failed
@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
def test_mla_mtp_mask_none(dtype):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    q_head_num = 128
    block_size = 128
    q_seq_lens = [1, 4, 2, 1]
    context_lengths = [192, 193, 194, 195]
    test_param = MlaTestParam(q_head_num, 1, block_size, 512, 64, 128, q_seq_lens, context_lengths,
                              0.001, dtype, dtype, "MASK_NONE")
    run_test(test_param)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [ms.bfloat16])
@pytest.mark.parametrize("seq_len", [1024, 2048])
def test_mla_long_seq(dtype, seq_len):
    """
    Feature: test mla
    Description: test mla.
    Expectation: the result is correct
    """
    q_seq_lens = [1, 1]
    context_lengths = [32 * 1024, seq_len]
    test_param = MlaTestParam(32, 1, 128, 512, 64, 1024, q_seq_lens,
                              context_lengths, 0.001, dtype, dtype, "MASK_NONE")
    run_test(test_param)


# q_seq_lens = [16, 1] context_lengths = [2048, 1024] 32, 1, 128, 512, 64, 8096 "MASK_SPEC" failed
# q_seq_lens = [1, 1] context_lengths = [32, 16] 32, 1, 128, 512, 64, 8096 "MASK_SPEC" failed
# q_seq_lens = [1, 1] context_lengths = [32, 16] 32, 1, 128, 512, 64, 256 "MASK_NONE" failed  rtol=atol=0.01 pass
# q_seq_lens = [1, 1] context_lengths = [32, 16] 32, 1, 128, 512, 64, 128 "MASK_NONE" pass
# q_seq_lens = [1, 1] context_lengths = [192, 193] 32, 1, 128, 512, 64, 256 "MASK_NONE" pass
# q_seq_lens = [16, 1] context_lengths = [128, 16] 32, 1, 128, 512, 64, 1024 "MASK_SPEC" pass
# q_seq_lens = [32, 1] context_lengths = [128, 16] 32, 1, 128, 512, 64, 1024 "MASK_SPEC" 0.01 0.01 jingdu budui
# q_seq_lens = [64, 1] context_lengths = [128, 16] 32, 1, 128, 512, 64, 1024 "MASK_SPEC"  MTE ERROR
# @pytest.mark.level1
# @pytest.mark.platform_arm_ascend910b_training
# @pytest.mark.env_onecard
# @pytest.mark.parametrize('dtype', [ms.bfloat16])
# def test_mla_pc_cp(dtype):
#     """
#     Feature: test mla
#     Description: test mla.
#     Expectation: the result is correct
#     """
#     q_seq_lens = [64, 1]
#     context_lengths = [128, 16]
#     test_param = MlaTestParam(32, 1, 128, 512, 64, 1024, q_seq_lens,
#                               context_lengths, 0.001, dtype, dtype, "MASK_SPEC")
#     run_test(test_param)
