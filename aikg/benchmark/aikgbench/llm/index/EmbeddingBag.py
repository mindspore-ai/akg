import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs EmbeddingBag operation.
    """

    def __init__(self, scale_grad_by_freq=False, mode='sum', sparse=False, include_last_offset=False, padding_idx=-1):
        super(Model, self).__init__()
        self.scale_grad_by_freq = scale_grad_by_freq
        self.mode = mode
        self.sparse = sparse
        self.include_last_offset = include_last_offset
        self.padding_idx = padding_idx

    def forward(self, weight, indices, offsets, per_sample_weights=None):
        """
        Perform EmbeddingBag operation.

        Args:
            weight: Embedding weight tensor
            indices: Indices tensor
            offsets: Offsets tensor
            per_sample_weights: Per-sample weights tensor (optional)

        Returns:
            Embedding bag output tensor
        """
        # Use torch.nn.functional.embedding_bag for the operation
        output = torch.nn.functional.embedding_bag(
            indices,
            weight,
            offsets=offsets,
            max_norm=None,
            norm_type=2,
            scale_grad_by_freq=self.scale_grad_by_freq,
            mode=self.mode,
            sparse=self.sparse,
            per_sample_weights=per_sample_weights,
            include_last_offset=self.include_last_offset,
            padding_idx=self.padding_idx,
        )

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: weight = np.random.randn(9).reshape(3, 3), indices = np.random.randint(0, 3, size=6)
    num_weights = 3
    weight_shape = [3, 3]
    indices_shape = [6]

    weight = torch.randn(weight_shape, dtype=torch.float32)
    indices = torch.randint(0, num_weights, indices_shape, dtype=torch.int64)
    offsets = torch.tensor([0, 2, 4, 5], dtype=torch.int64)
    per_sample_weights = torch.ones(indices_shape, dtype=torch.float32)

    return [weight, indices, offsets, per_sample_weights]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [False, 'sum', False, False, 1]  # scale_grad_by_freq=False, mode='sum', sparse=False, include_last_offset=False, padding_idx=1
