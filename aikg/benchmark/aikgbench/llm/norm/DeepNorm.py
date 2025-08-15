import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs DeepNorm operation.
    Based on 
    """

    def __init__(self, alpha=0.3, epsilon=1e-6):
        super(Model, self).__init__()
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x, gx, beta, gamma):
        """
        Perform DeepNorm operation.

        Args:
            x: Input tensor
            gx: Gate tensor
            beta: Shift parameter tensor
            gamma: Scale parameter tensor

        Returns:
            Tuple of (mean, rstd, y) where:
            - mean is the mean of the normalized tensor
            - rstd is the reciprocal standard deviation
            - y is the normalized output tensor
        """
        # Apply alpha scaling and add gate
        x_add = x * self.alpha + gx

        # Compute mean and variance
        mean = x_add.mean(-1, keepdim=True)
        diff = x_add - mean
        variance = diff.pow(2).mean(-1, keepdim=True)

        # Compute reciprocal standard deviation
        rstd = torch.rsqrt(variance + self.epsilon)

        # Apply normalization
        output = gamma * diff * rstd + beta

        return mean, rstd, output


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use the same shapes as in gen_data.py
    batch_size, seq_len, hidden_size = 3, 1, 4

    # Generate input tensors
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     dtype=torch.float32).reshape(batch_size, seq_len, hidden_size)
    gx = torch.tensor([2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8],
                      dtype=torch.float32).reshape(batch_size, seq_len, hidden_size)

    # Generate beta and gamma parameters
    beta = torch.tensor([0, 1, 2, 3], dtype=torch.float32)
    gamma = torch.tensor([0, 1, 2, 3], dtype=torch.float32)

    return [x, gx, beta, gamma]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [0.3, 1e-6]  # alpha=0.3, epsilon=1e-6
