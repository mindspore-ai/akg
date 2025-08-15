import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs Mmad operation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, Bias):
        """
        Perform Mmad operation.

        Args:
            A: Left matrix tensor [M, K]
            B: Right matrix tensor [K, N]
            Bias: Bias tensor [N]

        Returns:
            Output tensor C = A * B + Bias
        """
        # Perform matrix multiplication
        output = torch.matmul(A, B)

        # Add bias to each row
        output = output + Bias

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from README: A(32, 32), B(32, 32), Bias(1, 32)
    M, K, N = 32, 32, 32

    # Generate input tensors (using float16 as specified)
    A = torch.randn(M, K, dtype=torch.float16)
    B = torch.randn(K, N, dtype=torch.float16)

    # Generate bias (using float as specified)
    Bias = torch.randn(1, N, dtype=torch.float32)

    return [A, B, Bias]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
