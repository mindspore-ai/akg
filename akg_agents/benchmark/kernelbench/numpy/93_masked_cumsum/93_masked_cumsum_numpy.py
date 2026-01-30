import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x, mask) -> np.ndarray:
        masked_input = x * mask
        return np.cumsum(masked_input, axis=1)


batch_size = 128
input_shape = (4000,)
dim = 1


def get_inputs():
    x = np.random.rand(batch_size, *input_shape).astype(np.float16)
    mask = np.random.randint(0, 2, x.shape).astype(bool)  # Random boolean mask
    return [x, mask]


def get_init_inputs():
    return [dim]

