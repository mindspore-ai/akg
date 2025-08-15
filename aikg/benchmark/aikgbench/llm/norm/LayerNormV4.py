import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs LayerNormV4 operation.
    Based on 
    """

    def __init__(self, epsilon=1e-5):
        super(Model, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, normalized_shape, gamma, beta):
        """
        Perform LayerNormV4 operation.

        Args:
            x: Input tensor
            normalized_shape: Shape of the normalized dimensions
            gamma: Scale parameter tensor
            beta: Shift parameter tensor

        Returns:
            Tuple of (output, mean, rstd) where:
            - output is the normalized tensor
            - mean is the mean of the normalized dimensions
            - rstd is the reciprocal standard deviation
        """
        # Use PyTorch's native layer norm
        output, mean, variance = torch.ops.aten.native_layer_norm(
            x, normalized_shape, gamma, beta, eps=self.epsilon
        )

        # Convert variance to rstd (reciprocal standard deviation)
        rstd = 1.0 / torch.sqrt(variance + self.epsilon)

        return output, mean, rstd


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use the same shapes as in gen_data.py
    batch_size, seq_len, hidden_size = 1, 2, 32

    # Generate input tensor
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)

    # Generate normalized_shape (same as gamma shape)
    normalized_shape = (hidden_size,)

    # Generate gamma and beta parameters
    gamma = torch.ones(hidden_size, dtype=torch.float32)
    beta = torch.zeros(hidden_size, dtype=torch.float32)

    return [x, normalized_shape, gamma, beta]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [1e-5]  # epsilon=1e-5
