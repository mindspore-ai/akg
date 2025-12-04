import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional

# ============================================================================
# SGLang参考信息
# ============================================================================
# 源文件：python/sglang/srt/layers/attention/fla/fused_sigmoid_gating_recurrent.py
# SGLang函数：fused_sigmoid_gating_delta_rule_update
# 实现类型：Triton kernel
# 功能：融合sigmoid门控计算与递归delta规则更新
# 测试文件：无
# 输入参考：根据源文件中的函数签名和hybrid_linear_attn_backend.py中的使用方式推断
# ============================================================================

# ============================================================================
# 以下是从SGLang直接复制的Triton Kernel实现
# ============================================================================

@triton.heuristics(
    {
        "USE_INITIAL_STATE": lambda args: args["h0_source"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    p_a = a + bos * HV + i_hv
    p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        b_h *= tl.exp(b_g)

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Update pointers for next timestep
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV

    # Store final state back to h0_source with bounds checking
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_k[:, None] * V
                + o_v[None, :]
            )
            tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 8)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)
    grid = (NK, NV, N * HV)

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o

# ============================================================================
# AIKGBench标准接口
# ============================================================================

class Model(nn.Module):
    """直接使用复制的Triton Kernel实现"""
    def __init__(self, softplus_beta: float = 1.0, softplus_threshold: float = 20.0, use_qk_l2norm_in_kernel: bool = False):
        super().__init__()
        self.softplus_beta = softplus_beta
        self.softplus_threshold = softplus_threshold
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    def forward(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        initial_state_source: torch.Tensor,
        initial_state_indices: torch.Tensor,
        scale: Optional[float] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=self.softplus_beta,
            softplus_threshold=self.softplus_threshold,
            q=q,
            k=k,
            v=v,
            b=b,
            initial_state_source=initial_state_source,
            initial_state_indices=initial_state_indices,
            scale=scale,
            use_qk_l2norm_in_kernel=self.use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )


class ModelSGLang(nn.Module):
    """SGLang PyTorch实现"""
    def __init__(self, softplus_beta: float = 1.0, softplus_threshold: float = 20.0, use_qk_l2norm_in_kernel: bool = False):
        super().__init__()
        self.softplus_beta = softplus_beta
        self.softplus_threshold = softplus_threshold
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    def forward(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        b: torch.Tensor,
        initial_state_source: torch.Tensor,
        initial_state_indices: torch.Tensor,
        scale: Optional[float] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 设置默认scale
        if scale is None:
            scale = k.shape[-1] ** -0.5
        else:
            assert scale > 0, "scale must be positive"
        
        # 获取输入张量的形状
        B, T, H, K = q.shape
        _, _, HV, V = v.shape
        N = B if cu_seqlens is None else len(cu_seqlens) - 1
        
        # 创建输出张量
        o = torch.zeros(B, T, HV, V, dtype=v.dtype, device=v.device)
        
        # 处理可变长度序列或固定长度序列
        if cu_seqlens is not None:
            for i_n in range(N):
                bos = cu_seqlens[i_n].item()
                eos = cu_seqlens[i_n + 1].item()
                seq_len = eos - bos
                
                # 获取当前序列的索引
                idx = initial_state_indices[i_n].item() if initial_state_indices is not None else -1
                if idx >= 0 and initial_state_source is not None:
                    # 初始化隐藏状态
                    hidden_state = initial_state_source[idx].clone()
                else:
                    # 初始化为零
                    hidden_state = torch.zeros(HV, K, V, dtype=torch.float32, device=q.device)
                
                # 处理每个时间步
                for t in range(seq_len):
                    # 获取当前时间步的张量
                    current_q = q[0, bos + t].float()  # [H, K]
                    current_k = k[0, bos + t].float()  # [H, K]
                    current_v = v[0, bos + t].float()  # [HV, V]
                    current_a = a[bos + t].float()     # [HV]
                    current_b = b[bos + t].float()     # [HV]
                    
                    # 计算门控参数
                    current_A_log = A_log.float()      # [HV]
                    current_dt_bias = dt_bias.float()  # [HV]
                    
                    # Compute g = -exp(A_log) * softplus(a + dt_bias)
                    x = current_a + current_dt_bias
                    beta_x = self.softplus_beta * x
                    # Apply softplus with numerical stability
                    softplus_x = torch.where(
                        beta_x <= self.softplus_threshold,
                        (1.0 / self.softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
                        x,
                    )
                    current_g = -torch.exp(current_A_log) * softplus_x  # [HV]
                    
                    # Compute beta = sigmoid(b)
                    current_beta = torch.sigmoid(current_b)  # [HV]
                    
                    # Apply L2 normalization if enabled
                    if self.use_qk_l2norm_in_kernel:
                        current_q = current_q / (torch.norm(current_q, dim=-1, keepdim=True) + 1e-6)
                        current_k = current_k / (torch.norm(current_k, dim=-1, keepdim=True) + 1e-6)
                    
                    # Apply scale
                    current_q = current_q * scale
                    
                    # Apply gating to hidden state: h *= exp(g)
                    hidden_state *= torch.exp(current_g).reshape(HV, 1, 1)
                    
                    # Process each value head
                    for i_hv in range(HV):
                        # Calculate corresponding attention head
                        i_h = i_hv // (HV // H)
                        
                        # Delta rule: v -= sum(h * k, dim=0)
                        kv_product = torch.matmul(current_k[i_h].unsqueeze(0), hidden_state[i_hv])  # [1, V]
                        v_residual = current_v[i_hv] - kv_product.squeeze(0)  # [V]
                        
                        # Apply beta gating: v *= beta
                        v_residual *= current_beta[i_hv]
                        
                        # Update hidden state: h += k[:, None] * v[None, :]
                        hidden_state[i_hv] += torch.outer(current_k[i_h], v_residual)
                        
                        # Compute output: o = sum(h * q, dim=0)
                        qh_product = torch.matmul(current_q[i_h].unsqueeze(0), hidden_state[i_hv])  # [1, V]
                        o[0, bos + t, i_hv] = qh_product.squeeze(0).to(o.dtype)
                
                # Store final state back to initial_state_source
                if idx >= 0 and initial_state_source is not None:
                    initial_state_source[idx] = hidden_state.to(initial_state_source.dtype)
        else:
            # 固定长度序列处理
            for i_n in range(B):
                # 获取当前序列的索引
                idx = initial_state_indices[i_n].item() if initial_state_indices is not None else -1
                if idx >= 0 and initial_state_source is not None:
                    # 初始化隐藏状态
                    hidden_state = initial_state_source[idx].clone()
                else:
                    # 初始化为零
                    hidden_state = torch.zeros(HV, K, V, dtype=torch.float32, device=q.device)
                
                # 处理每个时间步
                for t in range(T):
                    # 获取当前时间步的张量
                    current_q = q[i_n, t].float()  # [H, K]
                    current_k = k[i_n, t].float()  # [H, K]
                    current_v = v[i_n, t].float()  # [HV, V]
                    current_a = a[i_n * T + t].float()  # [HV]
                    current_b = b[i_n * T + t].float()  # [HV]
                    
                    # 计算门控参数
                    current_A_log = A_log.float()      # [HV]
                    current_dt_bias = dt_bias.float()  # [HV]
                    
                    # Compute g = -exp(A_log) * softplus(a + dt_bias)
                    x = current_a + current_dt_bias
                    beta_x = self.softplus_beta * x
                    # Apply softplus with numerical stability
                    softplus_x = torch.where(
                        beta_x <= self.softplus_threshold,
                        (1.0 / self.softplus_beta) * torch.log(1.0 + torch.exp(beta_x)),
                        x,
                    )
                    current_g = -torch.exp(current_A_log) * softplus_x  # [HV]
                    
                    # Compute beta = sigmoid(b)
                    current_beta = torch.sigmoid(current_b)  # [HV]
                    
                    # Apply L2 normalization if enabled
                    if self.use_qk_l2norm_in_kernel:
                        current_q = current_q / (torch.norm(current_q, dim=-1, keepdim=True) + 1e-6)
                        current_k = current_k / (torch.norm(current_k, dim=-1, keepdim=True) + 1e-6)
                    
                    # Apply scale
                    current_q = current_q * scale
                    
                    # Apply gating to hidden state: h *= exp(g)
                    hidden_state *= torch.exp(current_g).reshape(HV, 1, 1)
                    
                    # Process each value head
                    for i_hv in range(HV):
                        # Calculate corresponding attention head
                        i_h = i_hv // (HV // H)
                        
                        # Delta rule: v -= sum(h * k, dim=0)
                        kv_product = torch.matmul(current_k[i_h].unsqueeze(0), hidden_state[i_hv])  # [1, V]
                        v_residual = current_v[i_hv] - kv_product.squeeze(0)  # [V]
                        
                        # Apply beta gating: v *= beta
                        v_residual *= current_beta[i_hv]
                        
                        # Update hidden state: h += k[:, None] * v[None, :]
                        hidden_state[i_hv] += torch.outer(current_k[i_h], v_residual)
                        
                        # Compute output: o = sum(h * q, dim=0)
                        qh_product = torch.matmul(current_q[i_h].unsqueeze(0), hidden_state[i_hv])  # [1, V]
                        o[i_n, t, i_hv] = qh_product.squeeze(0).to(o.dtype)
                
                # Store final state back to initial_state_source
                if idx >= 0 and initial_state_source is not None:
                    initial_state_source[idx] = hidden_state.to(initial_state_source.dtype)
        
        return o


def get_inputs():
    """生成测试输入"""
    # 根据hybrid_linear_attn_backend.py中的使用方式设置参数
    B = 1  # batch size
    T = 16  # sequence length
    H = 4   # number of heads
    HV = 4  # number of value heads
    K = 64  # key dimension
    V = 64  # value dimension
    
    dtype = torch.float16
    
    # 输入张量
    A_log = torch.randn(HV, dtype=torch.float32)
    a = torch.randn(B*T, HV, dtype=torch.float32)
    dt_bias = torch.randn(HV, dtype=torch.float32)
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype)
    b = torch.randn(B*T, HV, dtype=torch.float32)
    
    # 初始状态
    initial_state_source = torch.randn(1, HV, K, V, dtype=torch.float32)
    initial_state_indices = torch.tensor([0], dtype=torch.long)
    
    return [A_log, a, dt_bias, q, k, v, b, initial_state_source, initial_state_indices]


def get_init_inputs():
    """生成初始化参数"""
    return [1.0, 20.0, True]  # softplus_beta, softplus_threshold, use_qk_l2norm_in_kernel