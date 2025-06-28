import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: np.ndarray) -> np.ndarray:
        pad_width = ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding), (self.padding, self.padding))
        x_padded = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)

        windows = np.lib.stride_tricks.sliding_window_view(x_padded, (self.kernel_size, self.kernel_size, self.kernel_size), axis=(2, 3, 4))
        windows = windows[..., ::self.stride, ::self.stride, ::self.stride, :, :, :]

        output = np.mean(windows, axis=(-3, -2, -1))
        return output


batch_size = 16
channels = 32
depth = 64
height = 64
width = 64
kernel_size = 3
stride = 2
padding = 1


def get_inputs():
    x = np.random.randn(batch_size, channels, depth, height, width).astype(np.float16)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding]

