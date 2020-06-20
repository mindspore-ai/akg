# Copyright 2019 Huawei Technologies Co., Ltd
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

"""dsl: matmul_mansch"""
import akg
import akg.tvm
from akg import backend as cce
from akg.backend.cce_build import build_config


c_dtype = "float%d" % cce.OUT_WIDTH
a_dtype = "float%d" % cce.INP_WIDTH
b_dtype = "float%d" % cce.WGT_WIDTH

# Data arrangement
# a set of data points consist of a Burst, burstSize = n points
# a set of bursts consist of      a Cube,  cubeSize = burstSize * burstSize

# number of data points in a burst on m, k, n axes
m_burst_size = cce.BLOCK_IN
k_burst_size = cce.BLOCK_REDUCE
n_burst_size = cce.BLOCK_OUT


def gemm_para_check(matrix_shape, l1_tiling_shape, l0_tiling_shape):
    m_value = matrix_shape[0]
    k_value = matrix_shape[1]
    n_value = matrix_shape[2]

    block_m = l1_tiling_shape[0]
    block_k = l1_tiling_shape[1]
    block_n = l1_tiling_shape[2]

    block_ml = l0_tiling_shape[0]
    block_kl = l0_tiling_shape[1]
    block_nl = l0_tiling_shape[2]

    # check shape
    if type(m_value) != int or type(n_value) != int or type(k_value) != int:
        raise RuntimeError(
            "type of input shape value should be int")
    if m_value <= 0 or n_value <= 0 or k_value <= 0:
        raise RuntimeError(
            "input shape should not be less than 0: actual (M, K, N) = (%d, %d, %d)" % (m_value, k_value, n_value))
    if m_value < 16 or n_value < 16 or k_value < 16:
        raise RuntimeError(
            "input shape M or K or N should not be less than 16: actual (M, K, N) = (%d, %d, %d)" \
            % (m_value, k_value, n_value))
    if (m_value % 16 != 0) or (n_value % 16 != 0) or (k_value % 16 != 0):
        raise RuntimeError(
            "input shape M or K or N should be multiple of 16: actual (M, K, N) = (%d, %d, %d)" \
            % (m_value, k_value, n_value))

    # check the block
    if type(block_m) != int or type(block_n) != int or type(block_k) != int:
        raise RuntimeError(
            "type of input block value should be int")
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise RuntimeError(
            "input block param should not be less than 0: actual (block_m, block_n, block_k) = (%d, %d, %d)" 
            % (block_m, block_n, block_k))
    if block_m > m_value or block_n > n_value or block_k > k_value:
        raise RuntimeError(
            "input block param should not be less than shape value: actual (block_m, block_n, block_k) = (%d, %d, %d)" \
            % (block_m, block_n, block_k))
    if (block_m % 16 != 0) or (block_n % 16 != 0) or (block_k % 16 != 0):
        raise RuntimeError(
            "input shape block_m or block_k or block_n should be multiple of 16: actual (block_m, block_k, block_n) = \
            (%d, %d, %d)" % (block_m, block_k, block_n))

    # check the block L0
    if type(block_ml) != int or type(block_nl) != int or type(block_kl) != int:
        raise RuntimeError(
            "type of input block value should be int")
    if block_ml <= 0 or block_nl <= 0 or block_kl <= 0:
        raise RuntimeError(
            "input block param should not be less than 0: actual (block_ml, block_nl, block_kl) = (%d, %d, %d)" % \
            (block_ml, block_nl, block_kl))
    if block_ml > block_m or block_nl > block_n or block_kl > block_k:
        raise RuntimeError(
            "input block param should not be less than blockL1 value: actual (block_ml, block_nl, block_kl) = \
            (%d, %d, %d)" % (block_ml, block_nl, block_kl))
    if (block_ml % 16 != 0) or (block_nl % 16 != 0) or (block_kl % 16 != 0):
        raise RuntimeError(
            "input shape block_ml or block_kl or block_nl should be multiple of 16: \
            actual (block_ml, block_kl, block_nl) = (%d, %d, %d)" % (block_ml, block_kl, block_nl))


def gemm_dsl(matrix_shape, l1_tiling_shape, l0_tiling_shape, kernel_name='gemm_normal'):
    # param check
    gemm_para_check(matrix_shape, l1_tiling_shape, l0_tiling_shape)

    # get matrix axis shapes
    m_shape = matrix_shape[0]
    k_shape = matrix_shape[1]
    n_shape = matrix_shape[2]

    m_l1_tshape = l1_tiling_shape[0]
    k_l1_tshape = l1_tiling_shape[1]
    n_l1_tshape = l1_tiling_shape[2]

    m_l0_tshape = l0_tiling_shape[0]
    # not support kL0 tile yet, must keep kL0Tshape == kL1Tshape
    k_l0_tshape = l0_tiling_shape[1]
    n_l0_tshape = l0_tiling_shape[2]

    # compute matrix shape as cube
    a_shape = (m_shape // m_burst_size, k_shape // k_burst_size, m_burst_size, k_burst_size)
    b_shape = (k_shape // k_burst_size, n_shape // n_burst_size, n_burst_size, k_burst_size)
    c_shape = (n_shape // n_burst_size, m_shape // m_burst_size, m_burst_size, n_burst_size)

    # define placeholders
    a_value = akg.tvm.placeholder(a_shape, name='A', dtype=a_dtype)
    b_value = akg.tvm.placeholder(b_shape, name='B', dtype=b_dtype)

    # define reduce axis
    # kBurstAxis
    kb = akg.tvm.reduce_axis((0, k_shape // k_burst_size), name='kb')
    # kPointAxis
    kp = akg.tvm.reduce_axis((0, k_burst_size), name='kp')

    # define compute
    c_value = akg.tvm.compute(c_shape, lambda nb, mb, mp, np: akg.tvm.sum(
        a_value[mb, kb, mp, kp].astype(c_dtype) *
        b_value[kb, nb, np, kp].astype(c_dtype),
        axis=[kb, kp]),
        name='C')

    def gemm_schedule():
        # schedule
        s = akg.tvm.create_schedule(c_value.op)

        # compute L0 tiling params
        ml0_tile = m_l0_tshape // m_burst_size
        kl0_tile = k_l0_tshape // k_burst_size
        nl0_tile = n_l0_tshape // n_burst_size

        # compute L1 to L0 tiling params
        ml1_tile = (m_l1_tshape + m_l0_tshape - 1) // m_l0_tshape
        kl1_tile = (k_l1_tshape + k_l0_tshape - 1) // k_l0_tshape
        nl1_tile = (n_l1_tshape + n_l0_tshape - 1) // n_l0_tshape

        # cache_read and cache_write
        a_l1 = s.cache_read(a_value, cce.scope_cbuf, [c_value])
        a_l0 = s.cache_read(a_l1, cce.scope_ca, [c_value])

        b_l1 = s.cache_read(b_value, cce.scope_cbuf, [c_value])
        b_l0 = s.cache_read(b_l1, cce.scope_cb, [c_value])

        c_ub = s.cache_write(c_value, cce.scope_ubuf)
        c_l0 = s.cache_write(c_ub, cce.scope_cc)

        def schedule_k_tiling():
            # print("schedule_K_tiling")
            # tiling L1 and L0 on C_L0
            c_l0_n_o, c_l0_n_i_i = s[c_l0].split(c_l0.op.axis[0], factor=nl0_tile)
            c_l0_m_o, c_l0_m_i_i = s[c_l0].split(c_l0.op.axis[1], factor=ml0_tile)
            c_l0_k_o, c_l0_k_i_i = s[c_l0].split(c_l0.op.reduce_axis[0], factor=kl0_tile)

            c_l0_n_o_o, c_l0_n_i = s[c_l0].split(c_l0_n_o, factor=nl1_tile)
            c_l0_m_o_o, c_l0_m_i = s[c_l0].split(c_l0_m_o, factor=ml1_tile)
            c_l0_k_o_o, c_l0_k_i = s[c_l0].split(c_l0_k_o, factor=kl1_tile)
            # |---N---|     |--------N-------|        |------------------Nz--------------------|
            # nGM, mGM, kGM, nL1Tile, mL1Tile, kL1Tile, nL0Tile, mL0Tile, mBurstSize, nBurstSize, kL0Tile, kBurstSize
            s[c_l0].reorder(c_l0_n_o_o, c_l0_m_o_o, c_l0_k_o_o, c_l0_n_i, c_l0_m_i, 
                            c_l0_k_i, c_l0_n_i_i, c_l0_m_i_i, c_l0.op.axis[2], c_l0.op.axis[3],
                            c_l0_k_i_i, c_l0.op.reduce_axis[1])

            s[a_l0].compute_at(s[c_l0], c_l0_k_i)
            s[b_l0].compute_at(s[c_l0], c_l0_k_i)

            s[a_l1].compute_at(s[c_l0], c_l0_k_o_o)
            s[b_l1].compute_at(s[c_l0], c_l0_k_o_o)

            # tiling C_UB
            c_ub_n_o, c_ub_n_i_i = s[c_ub].split(c_ub.op.axis[0], factor=nl0_tile)
            c_ub_m_o, c_ub_m_i_i = s[c_ub].split(c_ub.op.axis[1], factor=ml0_tile)
            c_ub_n_o_o, c_ub_n_i = s[c_ub].split(c_ub_n_o, factor=nl1_tile)
            c_ub_m_o_o, c_ub_m_i = s[c_ub].split(c_ub_m_o, factor=ml1_tile)
            # |---N---| |--------N------| |------------------Nz------------------|
            # nGM, mGM, nL1Tile, mL1Tile, nL0Tile, mL0Tile, mBurstSize, nBurstSize
            s[c_ub].reorder(c_ub_n_o_o, c_ub_m_o_o, c_ub_n_i, c_ub_m_i, c_ub_n_i_i, c_ub_m_i_i, 
                            c_ub.op.axis[2], c_ub.op.axis[3])

            s[c_l0].compute_at(s[c_ub], c_ub_m_o_o)

            # tiling C
            c_n_o, c_n_i_i = s[c_value].split(c_value.op.axis[0], factor=nl0_tile)
            c_m_o, c_m_i_i = s[c_value].split(c_value.op.axis[1], factor=ml0_tile)
            c_n_o_o, c_n_i = s[c_value].split(c_n_o, factor=nl1_tile)
            c_m_o_o, c_m_i = s[c_value].split(c_m_o, factor=ml1_tile)
            # |---N---| |--------N------| |------------------Nz------------------|
            # nGM, mGM, nL1Tile, mL1Tile, nL0Tile, mL0Tile, mBurstSize, nBurstSize
            s[c_value].reorder(c_n_o_o, c_m_o_o, c_n_i, c_m_i, c_n_i_i, c_m_i_i, 
                               c_value.op.axis[2], c_value.op.axis[3])

            s[c_ub].compute_at(s[c_value], c_m_o_o)

            # emit_insn
            s[a_l1].emit_insn(a_l1.op.axis[0], 'dma_copy')
            s[b_l1].emit_insn(b_l1.op.axis[0], 'dma_copy')

            s[a_l0].emit_insn(a_l0.op.axis[0], 'dma_copy')
            s[b_l0].emit_insn(b_l0.op.axis[0], 'dma_copy')

            # mad_pattern value: 0 for gemm, 1 for convolution
            s[c_l0].pragma(c_l0_n_i_i, 'mad_pattern', 0)
            s[c_l0].emit_insn(c_l0_n_i_i, 'mad')
            s[c_ub].emit_insn(c_ub_n_i_i, 'dma_copy')
            s[c_value].emit_insn(c_n_i_i, 'dma_copy')

            s[c_l0].pragma(c_l0_k_o_o, 'is_reduce_k_outer', 1)
            s[c_l0].pragma(c_l0_k_i, 'is_reduce_k_outer', 1)

        def schedule_mn_tiling():
            # print("schedule_MN_tiling")
            # process L1 tile of A and B for computing C
            # GM -> L1
            nbc_value, mbc_value, mpc_value, npc_value = s[c_value].op.axis
            m_gm, m_ub = s[c_value].split(mbc_value, factor=ml0_tile)
            n_gm, n_ub = s[c_value].split(nbc_value, factor=nl0_tile)
            s[c_value].reorder(n_gm, m_gm, n_ub, m_ub, mpc_value, npc_value)
            store_pt_c = n_ub

            # attach AL1, BL1, C_UB to target loop.
            # notice that A_L0 in inner loop, for its continunous data
            s[c_ub].compute_at(s[c_value], m_gm)
            s[a_l1].compute_at(s[c_value], m_gm)
            s[b_l1].compute_at(s[c_value], n_gm)

            # process L0 tile of AL1, BL1 for computing C_UB
            nb_ub, mb_ub, mp_ub, np_ub = s[c_ub].op.axis
            m_ub, m_l0 = s[c_ub].split(mb_ub, factor=ml0_tile)
            n_ub, n_l0 = s[c_ub].split(nb_ub, factor=nl0_tile)
            s[c_ub].reorder(n_ub, m_ub, n_l0, m_l0, mp_ub, np_ub)  # ? ? 2 2 16 16
            store_pt_ub = n_l0

            # attach C_L0 to target loop.
            s[c_l0].compute_at(s[c_ub], m_ub)

            # split the reduce axis
            # L0_CUT3
            nb_l0, mb_l0, mp_l0, np_l0 = s[c_l0].op.axis
            k_gm, k_l0 = s[c_l0].split(kb, kl0_tile)
            s[c_l0].reorder(k_gm, nb_l0, mb_l0, mp_l0, np_l0, k_l0, kp)  # 4 2 2 16 16 4 16

            s[a_l0].compute_at(s[c_l0], k_gm)
            s[b_l0].compute_at(s[c_l0], k_gm)

            # emit_insn
            s[a_l1].emit_insn(s[a_l1].op.axis[0], 'dma_copy')
            s[b_l1].emit_insn(s[b_l1].op.axis[0], 'dma_copy')

            s[a_l0].emit_insn(s[a_l0].op.axis[0], 'dma_copy')
            s[b_l0].emit_insn(s[b_l0].op.axis[0], 'dma_copy')

            # mad_pattern value: 0 for gemm, 1 for convolution
            s[c_l0].pragma(nb_l0, 'mad_pattern', 0)
            s[c_l0].emit_insn(nb_l0, 'mad')

            s[c_ub].emit_insn(store_pt_ub, 'dma_copy')
            s[c_value].emit_insn(store_pt_c, 'dma_copy')
            s[c_l0].pragma(k_gm, 'is_reduce_k_outer', 1)

        if m_l0_tshape == m_l1_tshape and n_l0_tshape == n_l1_tshape:
            schedule_k_tiling()
        elif k_l0_tshape == k_l1_tshape:
            schedule_mn_tiling()

        with build_config:
            mod = akg.build(s, [a_value, b_value, c_value], "cce", name=kernel_name)
            return mod

    return gemm_schedule()
