try:
    from akg_agents.op.utils.triton_autotune_patch import apply_triton_patches
    apply_triton_patches()
except ImportError:
    pass
import triton
import triton.language as tl
import torch
import torch.nn as nn


@triton.jit
def symmetric_AA_kernel(
    A_ptr, A2_ptr,
    B, M,
    stride_Ab, stride_Am, stride_An,
    stride_A2b, stride_A2m, stride_A2n,
    CORE_NUM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Compute A2 = A @ A where A is [B, M, M] symmetric and A2 is also symmetric.
    Only computes upper triangle + diagonal (bm <= bn),
    mirrors to lower triangle via transposed store.
    Uses tiling BLOCK_M=128, BLOCK_K=256, BLOCK_N=256.
    Row/col block ratio: BLOCKS_PER_BN = BLOCK_N // BLOCK_M.
    Enumerate upper-triangle (bm, bn) blocks only; mirror when bm < bn * BLOCKS_PER_BN.
    """
    BLOCKS_PER_BN: tl.constexpr = BLOCK_N // BLOCK_M
    NUM_BLOCKS_M = tl.cdiv(M, BLOCK_M)
    NUM_BLOCKS_N = tl.cdiv(M, BLOCK_N)

    UPPER_BLOCKS_PER_BATCH = 0
    for bn_cand in range(NUM_BLOCKS_N):
        row_blocks = (bn_cand + 1) * BLOCKS_PER_BN
        row_blocks = tl.minimum(row_blocks, NUM_BLOCKS_M)
        UPPER_BLOCKS_PER_BATCH = UPPER_BLOCKS_PER_BATCH + row_blocks

    TOTAL_BLOCKS = B * UPPER_BLOCKS_PER_BATCH
    pid = tl.program_id(0)

    for block_idx in range(pid, TOTAL_BLOCKS, CORE_NUM):
        b = block_idx // UPPER_BLOCKS_PER_BATCH
        local_idx = block_idx % UPPER_BLOCKS_PER_BATCH

        offset = 0
        bn = 0
        bm = 0
        for bn_cand in range(NUM_BLOCKS_N):
            row_blocks = (bn_cand + 1) * BLOCKS_PER_BN
            row_blocks = tl.minimum(row_blocks, NUM_BLOCKS_M)
            in_this_bn = (local_idx >= offset) & (local_idx < offset + row_blocks)
            bn = tl.where(in_this_bn, bn_cand, bn)
            bm = tl.where(in_this_bn, local_idx - offset, bm)
            offset = offset + row_blocks

        m_off_i = bm * BLOCK_M + tl.arange(0, BLOCK_M)
        n_off_j = bn * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_i = m_off_i < M
        mask_j = n_off_j < M

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for k in range(0, M, BLOCK_K):
            k_off = k + tl.arange(0, BLOCK_K)
            k_mask = k_off < M

            a_left = tl.load(
                A_ptr + b * stride_Ab
                + m_off_i[:, None] * stride_Am
                + k_off[None, :] * stride_An,
                mask=mask_i[:, None] & k_mask[None, :], other=0.0
            )

            a_right = tl.load(
                A_ptr + b * stride_Ab
                + k_off[:, None] * stride_Am
                + n_off_j[None, :] * stride_An,
                mask=k_mask[:, None] & mask_j[None, :], other=0.0
            )

            acc += tl.dot(a_left, a_right, allow_tf32=False)

        store_mask = mask_i[:, None] & mask_j[None, :]

        tl.store(
            A2_ptr + b * stride_A2b
            + m_off_i[:, None] * stride_A2m
            + n_off_j[None, :] * stride_A2n,
            acc, mask=store_mask
        )

        if bm < bn * BLOCKS_PER_BN:
            acc_lo = tl.extract_slice(acc, (0, 0), (BLOCK_M, BLOCK_M), (1, 1))
            n_off_lo = bn * BLOCK_N + tl.arange(0, BLOCK_M)
            mask_j_lo = n_off_lo < M
            tl.store(
                A2_ptr + b * stride_A2b
                + n_off_lo[:, None] * stride_A2m
                + m_off_i[None, :] * stride_A2n,
                acc_lo.trans(),
                mask=mask_j_lo[:, None] & mask_i[None, :]
            )

            acc_hi = tl.extract_slice(acc, (0, BLOCK_M), (BLOCK_M, BLOCK_M), (1, 1))
            n_off_hi = bn * BLOCK_N + BLOCK_M + tl.arange(0, BLOCK_M)
            mask_j_hi = n_off_hi < M
            tl.store(
                A2_ptr + b * stride_A2b
                + n_off_hi[:, None] * stride_A2m
                + m_off_i[None, :] * stride_A2n,
                acc_hi.trans(),
                mask=mask_j_hi[:, None] & mask_i[None, :]
            )
        elif bm == bn * BLOCKS_PER_BN and BLOCKS_PER_BN > 1:
            acc_hi = tl.extract_slice(acc, (0, BLOCK_M), (BLOCK_M, BLOCK_M), (1, 1))
            m_off_mirror = (bm + 1) * BLOCK_M + tl.arange(0, BLOCK_M)
            n_off_mirror = bn * BLOCK_N + tl.arange(0, BLOCK_M)
            mask_i_mirror = m_off_mirror < M
            mask_j_mirror = n_off_mirror < M
            tl.store(
                A2_ptr + b * stride_A2b
                + m_off_mirror[:, None] * stride_A2m
                + n_off_mirror[None, :] * stride_A2n,
                acc_hi.trans(),
                mask=mask_i_mirror[:, None] & mask_j_mirror[None, :]
            )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        try:
            device = torch.npu.current_device()
            properties = triton.runtime.driver.active.utils.get_device_properties(device)
            self.CUBE_CORE_NUM = properties.get("num_aicore", 20)
        except Exception:
            self.CUBE_CORE_NUM = 20

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        squeeze_back = A.ndim == 2
        if squeeze_back:
            A = A.unsqueeze(0)
        if not A.is_contiguous():
            A = A.contiguous()

        B, M, _ = A.shape
        A2 = torch.empty((B, M, M), dtype=A.dtype, device=A.device)

        BLOCK_M = 128
        BLOCK_K = 256
        BLOCK_N = 256

        grid = (self.CUBE_CORE_NUM,)
        symmetric_AA_kernel[grid](
            A, A2,
            B, M,
            A.stride(0), A.stride(1), A.stride(2),
            A2.stride(0), A2.stride(1), A2.stride(2),
            CORE_NUM=self.CUBE_CORE_NUM,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
        )

        if squeeze_back:
            A2 = A2.squeeze(0)
        return A2