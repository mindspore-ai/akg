"""HyperConnectionOutputCell base，遵循 akg_kernels_bench/dynamic_shape 格式。"""

from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore

RATE = 4
HIDDEN_SIZE = 3584
PACKED_HIDDEN_SIZE = RATE * HIDDEN_SIZE

DEFAULT_INPUT_DTYPE = torch.bfloat16
DEFAULT_INPUT_SEED = 2028

BATCH_SIZE = 2


class Model(nn.Module):
    """PyTorch reference for HyperConnectionOutputCell.construct()."""

    def __init__(self, rate=RATE, hidden_size=HIDDEN_SIZE, input_dtype=DEFAULT_INPUT_DTYPE):
        super().__init__()
        self.rate = rate
        self.hidden_size = hidden_size
        self.input_dtype = input_dtype

    def forward(self, h_res, h_post, original_streams, sublayer_out):
        seq_len, batch_size, packed_hidden_size = original_streams.shape
        assert packed_hidden_size == self.rate * self.hidden_size

        x_streams = original_streams.float().view(
            seq_len, batch_size, self.rate, self.hidden_size
        )
        residual_part = torch.matmul(h_res.float(), x_streams)
        post_part = h_post.float() * sublayer_out.float().view(
            seq_len, batch_size, 1, self.hidden_size
        )
        updated = residual_part + post_part
        return updated.reshape(seq_len, batch_size, packed_hidden_size).to(
            self.input_dtype
        )


def _sinkhorn_knopp(logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    logits = logits.float()
    logits_max = logits.amax(dim=-1, keepdim=True)
    matrix = torch.exp(logits - logits_max)
    for _ in range(iters):
        matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + eps)
        matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + eps)
    return matrix


def _make_case_inputs(seq_len: int) -> list[torch.Tensor]:
    torch.manual_seed(DEFAULT_INPUT_SEED + int(seq_len))

    h_res_logits = torch.randn(
        int(seq_len), BATCH_SIZE, RATE, RATE, dtype=torch.float32
    )
    h_res = _sinkhorn_knopp(h_res_logits, 20, 1e-6).to(DEFAULT_INPUT_DTYPE)

    h_post = (
        2.0
        * torch.sigmoid(
            torch.randn(int(seq_len), BATCH_SIZE, RATE, 1, dtype=torch.float32)
        )
    ).to(DEFAULT_INPUT_DTYPE)

    original_streams = torch.randn(
        int(seq_len), BATCH_SIZE, PACKED_HIDDEN_SIZE, dtype=torch.float32
    ).to(DEFAULT_INPUT_DTYPE)
    sublayer_out = torch.randn(
        int(seq_len), BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32
    ).to(DEFAULT_INPUT_DTYPE)

    return [h_res, h_post, original_streams, sublayer_out]


def get_init_inputs():
    return [RATE, HIDDEN_SIZE, DEFAULT_INPUT_DTYPE]


_DYN_SEQ_LENS = (4096, 8192, 12288, 16384, 24576, 32768)


def get_inputs_dyn_list():
    return [_make_case_inputs(seq_len) for seq_len in _DYN_SEQ_LENS]
