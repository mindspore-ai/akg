import torch
import triton
import triton.language as tl

@triton.jit
def _attn_fwd(Q, K, V, M, Out, acc, sm_scale,
              stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
              stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,
              stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,
              stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
              Z: tl.constexpr, H: tl.constexpr,
              N_CTX: tl.constexpr,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              ):
    # Total number of blocks in sequence dimension (M)
    NUM_BLOCKS_M = N_CTX // BLOCK_M
    # Total tasks = number of sequence blocks × batch size (Z) × number of attention heads (H)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    # Current M-dimension block index
    pid = tl.program_id(0)

    for block_idx in range(pid, NUM_BLOCKS, 20):
        task_hz_idx = block_idx // NUM_BLOCKS_M
        task_m_idx = block_idx % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
        # Create block pointers for Q, K, V, Output
        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        # Initialize offsets
        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0

        # Initialize accumulator
        acc_ptr = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # load q: it will stay in SRAM throughout
        q = tl.load(Q_block_ptr)

        lo, hi = 0, N_CTX  # Process the entire context

        # Adjust K and V block pointers to the starting position `lo`
        K_block_ptr = tl.advance(K_block_ptr, (lo, 0))  # K is [HEAD_DIM, N_CTX], shift along the second dim by lo
        V_block_ptr = tl.advance(V_block_ptr, (lo, 0))  # V is [N_CTX, HEAD_DIM], shift along the first dim by lo

        # Iterate over all k, v blocks in the current stage and accumulate the output
        for start_n in range(lo, hi, BLOCK_N):  # Process BLOCK_N columns at a time
            start_n = tl.multiple_of(start_n, BLOCK_N)  # Align column start position
            # -- Compute qk ----
            k = tl.load(K_block_ptr)
            # Modify K
            trans_k = tl.trans(k)
            qk = tl.dot(q, trans_k)

            qk = qk * sm_scale
            m_ij = tl.maximum(m_i, tl.max(qk, 1))  # Scaled max
            qk = qk - m_ij[:, None]  # Stabilize

            # Softmax weights p = exp(qk)
            p = tl.math.exp(qk)
            p_cast = p.to(k.dtype)

            v = tl.load(V_block_ptr)  # Load corresponding V block
            pv = tl.dot(p_cast, v)
            l_ij = tl.sum(p, 1)  # Softmax denominator (sum of each row)
            # -- Update m_i and l_i
            alpha = tl.math.exp(m_i - m_ij)  # Update factor: exp difference between old and new max
            l_i = l_i * alpha + l_ij  # Update softmax denominator
            # -- Update output accumulator --
            acc_ptr = acc_ptr * alpha[:, None]
            acc_ptr = tl.dot(p_cast, v, acc_ptr)

            m_i = m_ij  # Update current block max
            # Advance V and K block pointers to next BLOCK_N range
            V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
            K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

        m_i += tl.math.log(l_i)
        accumulator = acc_ptr / l_i[:, None]

        m_ptrs = M + task_hz_idx * N_CTX + offs_m

        tl.store(m_ptrs, m_i)
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))

class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale):
        super().__init__()
        self.sm_scale = sm_scale

    def forward(self, q, k, v):
        """
        Forward computation interface:
        Args:
            ctx: Context object
            q: Query tensor (Q), shape [Z, H, N_CTX, HEAD_DIM]
            k: Key tensor (K), shape [Z, H, N_CTX, HEAD_DIM]
            v: Value tensor (V), shape [Z, H, N_CTX, HEAD_DIM]
        Returns:
            o: Attention output tensor, shape [Z, H, N_CTX, HEAD_DIM]
        """
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}

        o = torch.empty_like(q)
        

        # Number of NPU cores (adjust based on hardware)
        num_cores = 20
        acc = torch.zeros((q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K), dtype=torch.float32, device=q.device)
        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

        BM, BN = 64, 128

        _attn_fwd[(num_cores,)](
            q, k, v, M, o, acc, self.sm_scale,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1], N_CTX=q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            BLOCK_M=BM,
            BLOCK_N=BN,
        )
        return o