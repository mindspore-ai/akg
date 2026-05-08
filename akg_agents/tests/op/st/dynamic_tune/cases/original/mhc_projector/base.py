"""mHC projector base，遵循 akg_kernels_bench/dynamic_shape 格式。"""

from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore

RATE = 4
HIDDEN_SIZE = 3584
PACKED_HIDDEN_SIZE = RATE * HIDDEN_SIZE

DEFAULT_INPUT_DTYPE = torch.bfloat16
DEFAULT_INIT_GATING_FACTOR = 0.01
DEFAULT_EXPAND_POST = 2.0
DEFAULT_WEIGHT_SEED = 2026
DEFAULT_INPUT_SEED = 2027

BATCH_SIZE = 2


class Model(nn.Module):
    """Packed hidden_states -> h_pre, h_post, h_res_logits."""

    def __init__(
        self,
        rate=RATE,
        hidden_size=HIDDEN_SIZE,
        init_gating_factor=DEFAULT_INIT_GATING_FACTOR,
        expand_post=DEFAULT_EXPAND_POST,
        input_dtype=DEFAULT_INPUT_DTYPE,
        seed=DEFAULT_WEIGHT_SEED,
    ):
        super().__init__()
        self.rate = rate
        self.hidden_size = hidden_size
        self.expand_post = expand_post
        self.input_dtype = input_dtype
        self.eps = 1e-6

        dim = rate + rate + rate * rate
        torch.manual_seed(seed)

        mapping_weight = (
            torch.randn(rate * hidden_size, dim, dtype=torch.float32) * 1e-4
        )
        self.register_buffer("mapping_weight", mapping_weight)

        alpha = torch.full((1, 1, 1, 1), init_gating_factor, dtype=torch.float32)
        self.register_buffer("alpha_pre", alpha.clone())
        self.register_buffer("alpha_post", alpha.clone())
        self.register_buffer("alpha_res", alpha.clone())

        pre_bias = torch.full(
            (1, 1, 1, rate),
            -torch.log(torch.tensor(3.0)),
            dtype=torch.float32,
        )
        post_bias = torch.zeros((1, 1, 1, rate), dtype=torch.float32)
        res_bias = (
            (torch.eye(rate, dtype=torch.float32) - 1.0) * 5.0
        ).reshape(1, 1, 1, rate * rate)
        self.register_buffer("bias_pre", pre_bias)
        self.register_buffer("bias_post", post_bias)
        self.register_buffer("bias_res", res_bias)

    def forward(self, hidden_states):
        seq_len, batch_size, packed_hidden_size = hidden_states.shape
        assert packed_hidden_size == self.rate * self.hidden_size

        hidden_states_fp32 = hidden_states.float()
        norm_x = hidden_states_fp32 * torch.rsqrt(
            hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )

        projected = torch.matmul(norm_x, self.mapping_weight).view(
            seq_len, batch_size, 1, -1
        )
        h_pre_raw = projected[..., : self.rate]
        h_post_raw = projected[..., self.rate : 2 * self.rate]
        h_res_raw = projected[..., 2 * self.rate :]

        h_pre = torch.sigmoid(self.alpha_pre * h_pre_raw + self.bias_pre)
        h_post = self.expand_post * torch.sigmoid(
            self.alpha_post * h_post_raw + self.bias_post
        )
        h_res_logits = (
            self.alpha_res * h_res_raw + self.bias_res
        ).view(seq_len, batch_size, self.rate, self.rate)

        return (
            h_pre.to(self.input_dtype),
            h_post.view(seq_len, batch_size, self.rate, 1).to(self.input_dtype),
            h_res_logits,
        )


def _make_hidden_states(seq_len: int) -> torch.Tensor:
    torch.manual_seed(DEFAULT_INPUT_SEED + int(seq_len))
    return torch.randn(
        int(seq_len),
        BATCH_SIZE,
        PACKED_HIDDEN_SIZE,
        dtype=torch.float32,
    ).to(DEFAULT_INPUT_DTYPE)


def get_init_inputs():
    return [
        RATE,
        HIDDEN_SIZE,
        DEFAULT_INIT_GATING_FACTOR,
        DEFAULT_EXPAND_POST,
        DEFAULT_INPUT_DTYPE,
        DEFAULT_WEIGHT_SEED,
    ]


_DYN_SEQ_LENS = (4096, 8192, 12288, 16384, 24576, 32768, 65536)


def get_inputs_dyn_list():
    return [[_make_hidden_states(seq_len)] for seq_len in _DYN_SEQ_LENS]
