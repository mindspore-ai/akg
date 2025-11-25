import numpy as np

class Model:
    """
    Simple model that performs Max Pooling 3D.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False, ceil_mode: bool = False):
        """
        Initializes the Max Pooling 3D layer.

        Args:
            kernel_size (int): Size of the kernel for the max pooling operation.
            stride (int, optional): Stride of the pooling operation. Defaults to None, which means stride is equal to kernel_size.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
            dilation (int, optional): Spacing between kernel elements. Defaults to 1.
            return_indices (bool, optional): Whether to return indices of the maximum values. Defaults to False.
            ceil_mode (bool, optional): When True, the output size is ceil(input_size / stride) instead of floor. Defaults to False.
        """
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        batch_size, channels, depth, height, width = x.shape

        # 三维padding处理
        padded_input = np.pad(x,
                              pad_width=((0,0), (0,0), 
                                       (self.padding, self.padding),
                                       (self.padding, self.padding),
                                       (self.padding, self.padding)),
                              mode='constant',
                              constant_values=-np.inf)

        # 计算输出尺寸
        output_d = (depth + 2*self.padding - self.dilation*(self.kernel_size-1) -1) // self.stride + 1
        output_h = (height + 2*self.padding - self.dilation*(self.kernel_size-1) -1) // self.stride + 1
        output_w = (width + 2*self.padding - self.dilation*(self.kernel_size-1) -1) // self.stride + 1

        output = np.zeros((batch_size, channels, output_d, output_h, output_w))

        # 五层循环处理三维卷积
        for b in range(batch_size):
            for c in range(channels):
                vol = padded_input[b, c]
                for d in range(output_d):
                    d_start = d * self.stride
                    for h in range(output_h):
                        h_start = h * self.stride
                        for w in range(output_w):
                            w_start = w * self.stride

                            max_val = -np.inf
                            # 三维滑动窗口
                            for kd in range(self.kernel_size):
                                d_idx = d_start + kd * self.dilation
                                if d_idx >= vol.shape[0]:
                                    continue
                                for kh in range(self.kernel_size):
                                    h_idx = h_start + kh * self.dilation
                                    if h_idx >= vol.shape[1]:
                                        continue
                                    for kw in range(self.kernel_size):
                                        w_idx = w_start + kw * self.dilation
                                        if w_idx >= vol.shape[2]:
                                            continue
                                        val = vol[d_idx, h_idx, w_idx]
                                        if val > max_val:
                                            max_val = val
                            output[b, c, d, h, w] = max_val
        return output

batch_size = 16
channels = 32
dim1 = 64
dim2 = 64
dim3 = 64
kernel_size = 3
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = np.random.rand(batch_size, channels, dim1, dim2, dim3).astype(np.float16)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]