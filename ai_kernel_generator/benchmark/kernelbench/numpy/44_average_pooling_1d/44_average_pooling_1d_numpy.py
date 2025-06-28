import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Apply padding
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding))
        x_padded = np.pad(x, pad_width, mode='constant', constant_values=0)

        # Calculate sliding window view
        from numpy.lib.stride_tricks import sliding_window_view
        window_view = sliding_window_view(x_padded, self.kernel_size, axis=2)
        window_view = window_view[:, :, ::self.stride, :]

        # Compute mean over kernel window
        output = np.mean(window_view, axis=-1)
        
        return output


batch_size = 16
in_channels = 32
input_length = 128
kernel_size = 4
stride = 2
padding = 1


def get_inputs():
    x = np.random.randn(batch_size, in_channels, input_length).astype(np.float16)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]

