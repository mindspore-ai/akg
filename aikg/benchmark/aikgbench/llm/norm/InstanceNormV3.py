import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs InstanceNormV3 operation.
    Based on 
    """

    def __init__(self, epsilon=1e-5, data_format="NCHW"):
        super(Model, self).__init__()
        self.epsilon = epsilon
        self.data_format = data_format

    def forward(self, x, gamma, beta):
        """
        Perform InstanceNormV3 operation.

        Args:
            x: Input tensor
            gamma: Scale parameter tensor
            beta: Shift parameter tensor

        Returns:
            Tuple of (output, mean, variance) where:
            - output is the normalized tensor
            - mean is the mean of each instance
            - variance is the variance of each instance
        """
        # Determine reduction axes based on data format
        if self.data_format == 'NHWC':
            reduce_axis = [1, 2]
            gamma = gamma.reshape([1, 1, 1, gamma.shape[0]])
            beta = beta.reshape([1, 1, 1, beta.shape[0]])
        else:  # NCHW
            reduce_axis = [2, 3]
            gamma = gamma.reshape([1, gamma.shape[0], 1, 1])
            beta = beta.reshape([1, beta.shape[0], 1, 1])

        # Compute mean and variance
        mean = torch.mean(x, dim=reduce_axis, keepdim=True)
        var = torch.mean(torch.pow((x - mean), 2),
                         dim=reduce_axis, keepdim=True)

        # Compute reciprocal standard deviation
        rstd = 1 / torch.sqrt(var + self.epsilon)

        # Apply normalization
        tmp_x = (x - mean) * rstd
        output = tmp_x * gamma + beta

        return output, mean, var


def get_inputs():
    """
    Generate random input tensors for testing.
    Based on 
    """
    # Use the same shapes as in gen_data.py
    batch_size, channels, height, width = 1, 8, 4, 4

    # Generate input tensor
    x = torch.ones(batch_size, channels, height,
                   width, dtype=torch.float32) * 0.77

    # Generate gamma and beta parameters
    gamma = torch.ones(channels, dtype=torch.float32) * 1.5
    beta = torch.ones(channels, dtype=torch.float32) * 0.5

    return [x, gamma, beta]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    Based on parameters
    """
    return [1e-5, "NCHW"]  # epsilon=1e-5, data_format="NCHW"
