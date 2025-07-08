import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, height, width = x.shape

        # Pad input with -inf
        padded_input = np.pad(x, 
                              pad_width=((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              mode='constant',
                              constant_values=-np.inf)

        # Calculate output dimensions
        output_h = (height + 2*self.padding - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1
        output_w = (width + 2*self.padding - self.dilation*(self.kernel_size-1) - 1) // self.stride + 1

        output = np.zeros((batch_size, channels, output_h, output_w))

        for b in range(batch_size):
            for c in range(channels):
                img = padded_input[b, c]
                for h in range(output_h):
                    h_start = h * self.stride
                    for w in range(output_w):
                        w_start = w * self.stride

                        max_val = -np.inf
                        for kh in range(self.kernel_size):
                            h_idx = h_start + kh * self.dilation
                            if h_idx >= img.shape[0]:
                                continue
                            for kw in range(self.kernel_size):
                                w_idx = w_start + kw * self.dilation
                                if w_idx >= img.shape[1]:
                                    continue
                                val = img[h_idx, w_idx]
                                if val > max_val:
                                    max_val = val
                        output[b, c, h, w] = max_val
        return output


batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3


def get_inputs():
    x = np.random.randn(batch_size, channels, height, width).astype(np.float16)
    return [x]


def get_init_inputs():
    return [kernel_size, stride, padding, dilation]

