import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs GemmV2 operation.
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, A, B, alpha, beta, C):
        """
        Perform GemmV2 operation.

        Args:
            A: First input tensor
            B: Second input tensor
            alpha: Alpha scaling factor
            beta: Beta scaling factor
            C: Third input tensor

        Returns:
            Output tensor: out = α(A @ B) + βC
        """
        # Perform matrix multiplication
        matmul_result = torch.matmul(A, B)

        # Apply alpha scaling
        scaled_result = alpha * matmul_result

        # Apply beta scaling to C and add
        output = scaled_result + beta * C

        return output


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Use shapes from README: A(2, 2), B(2, 2), C(2, 2)
    m, n, k = 2, 2, 2

    # Generate input tensors (using float16 as specified)
    A = torch.randn(m, k, dtype=torch.float16)
    B = torch.randn(k, n, dtype=torch.float16)

    # Generate scaling factors
    alpha = torch.randn(1, dtype=torch.float16)
    beta = torch.randn(1, dtype=torch.float16)

    # Generate C tensor (using float32 as specified)
    C = torch.randn(m, n, dtype=torch.float32)

    return [A, B, alpha, beta, C]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return []
