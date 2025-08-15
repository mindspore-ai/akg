import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs EmbeddingDenseGradV2 operation.
    """

    def __init__(self, num_weights=4, padding_idx=0, scale_grad_by_freq=False):
        super(Model, self).__init__()
        self.num_weights = num_weights
        self.padding_idx = padding_idx
        self.scale_grad_by_freq = scale_grad_by_freq

    def forward(self, grad, sort_indices, pos_idx, num_weights_tensor=None, padding_idx_tensor=None, scale_grad_by_freq_tensor=None):
        """
        Perform EmbeddingDenseGradV2 operation.

        Args:
            grad: Gradient tensor
            sort_indices: Sorted indices tensor
            pos_idx: Position indices tensor
            num_weights_tensor: Number of weights tensor (optional)
            padding_idx_tensor: Padding index tensor (optional)
            scale_grad_by_freq_tensor: Scale gradient by frequency tensor (optional)

        Returns:
            Embedding gradient tensor
        """
        # Use provided tensors or fall back to instance variables
        num_weights = num_weights_tensor.item(
        ) if num_weights_tensor is not None else self.num_weights
        padding_idx = padding_idx_tensor.item(
        ) if padding_idx_tensor is not None else self.padding_idx
        scale_grad_by_freq = scale_grad_by_freq_tensor.item(
        ) if scale_grad_by_freq_tensor is not None else self.scale_grad_by_freq

        # Use torch.ops.aten.embedding_dense_backward for the operation
        result = torch.ops.aten.embedding_dense_backward(
            grad_output=grad,
            indices=sort_indices,
            num_weights=num_weights,
            padding_idx=padding_idx,
            scale_grad_by_freq=scale_grad_by_freq
        )

        return result


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from gen_data.py: grad = np.random.randn(6).reshape(2, 3), num_weights = 4
    grad_shape = [2, 3]
    num_weights = 4

    grad = torch.randn(grad_shape, dtype=torch.float32)
    sort_indices = torch.randint(0, num_weights, (2,), dtype=torch.int32)
    pos_idx = torch.randint(0, 2, (2,), dtype=torch.int32)
    num_weights_tensor = torch.tensor([num_weights], dtype=torch.int32)
    padding_idx_tensor = torch.tensor([0], dtype=torch.int32)
    scale_grad_by_freq_tensor = torch.tensor([False], dtype=torch.bool)

    return [grad, sort_indices, pos_idx, num_weights_tensor, padding_idx_tensor, scale_grad_by_freq_tensor]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return [4, 0, False]  # num_weights=4, padding_idx=0, scale_grad_by_freq=False
