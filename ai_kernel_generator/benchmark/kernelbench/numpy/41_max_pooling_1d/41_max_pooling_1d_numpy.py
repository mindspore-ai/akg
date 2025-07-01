import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0,
                     dilation: int = 1, return_indices: bool = False):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices

    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size, features, seq_length = x.shape
        # Calculate output sequence length
        output_length = ((seq_length + 2*self.padding - self.dilation*(self.kernel_size-1) - 1) // self.stride) + 1
        # Apply self.padding
        padded_input = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding)), mode='constant', constant_values=-np.inf)
        # Initialize output
        output = np.zeros((batch_size, features, output_length))
        # Perform max pooling
        for b in range(batch_size):
            for f in range(features):
                for i in range(output_length):
                    start = i * self.stride
                    # Generate window indices with self.dilation
                    indices = [start + j*self.dilation for j in range(self.kernel_size)]
                    # Get window values
                    window = padded_input[b, f, indices]
                    # Store maximum value
                    output[b, f, i] = np.max(window)
        return output


batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False


def get_inputs():
    x = np.random.randn(batch_size, features, sequence_length).astype(np.float16)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]

