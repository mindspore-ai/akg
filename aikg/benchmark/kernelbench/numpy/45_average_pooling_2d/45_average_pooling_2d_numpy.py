import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, kernel_size: int, stride: int = 3, padding: int = 0):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def __call__(self, x: np.ndarray) -> np.ndarray:
        B, C, H, W = x.shape
        out_h = (H - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_w = (W - self.kernel_size + 2 * self.padding) // self.stride + 1

        shape = (B, C, out_h, out_w, self.kernel_size, self.kernel_size)
        strides = (
            x.strides[0],
            x.strides[1],
            self.stride * x.strides[2],
            self.stride * x.strides[3],
            x.strides[2],
            x.strides[3]
        )
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        output = np.mean(windows, axis=(-2, -1))
        return output


batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3


def get_inputs():
    x = np.random.randn(batch_size, channels, height, width).astype(np.float16)
    return [x]


def get_init_inputs():
    return [kernel_size]

