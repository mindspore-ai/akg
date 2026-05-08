"""mHC stream weighted sum base，遵循 akg_kernels_bench/dynamic_shape 格式。"""

from __future__ import annotations

import torch  # type: ignore
import torch.nn as nn  # type: ignore

RATE = 4
HIDDEN_SIZE = 8192
PACKED_HIDDEN_SIZE = RATE * HIDDEN_SIZE

DEFAULT_INPUT_DTYPE = torch.bfloat16
DEFAULT_INPUT_SEED = 2029

BATCH_SIZE = 2


class Model(nn.Module):
    """h_pre @ packed streams -> aggregated hidden state."""

    def __init__(self, rate=RATE, hidden_size=HIDDEN_SIZE, input_dtype=DEFAULT_INPUT_DTYPE):
        super().__init__()
        self.rate = rate
        self.hidden_size = hidden_size
        self.input_dtype = input_dtype

    def forward(self, h_pre, original_streams):
        seq_len, batch_size, packed_hidden_size = original_streams.shape
        assert packed_hidden_size == self.rate * self.hidden_size
        x_streams = original_streams.float().view(
            seq_len, batch_size, self.rate, self.hidden_size
        )
        aggregated = torch.matmul(h_pre.float(), x_streams).squeeze(2)
        return aggregated.to(self.input_dtype)


def _make_case_inputs(seq_len: int) -> list[torch.Tensor]:
    torch.manual_seed(DEFAULT_INPUT_SEED + int(seq_len))
    h_pre = torch.sigmoid(
        torch.randn(int(seq_len), BATCH_SIZE, 1, RATE, dtype=torch.float32)
    ).to(DEFAULT_INPUT_DTYPE)
    original_streams = torch.randn(
        int(seq_len), BATCH_SIZE, PACKED_HIDDEN_SIZE, dtype=torch.float32
    ).to(DEFAULT_INPUT_DTYPE)
    return [h_pre, original_streams]


def get_init_inputs():
    return [RATE, HIDDEN_SIZE, DEFAULT_INPUT_DTYPE]


_DYN_SEQ_LENS = (4096, 8192, 12288, 16384, 24576, 32768, 65536)


def get_inputs_dyn_list():
    return [_make_case_inputs(seq_len) for seq_len in _DYN_SEQ_LENS]
