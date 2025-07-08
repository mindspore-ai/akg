import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x) -> np.ndarray:
        flipped = np.flip(x, axis=self.dim)
        cum_result = np.cumsum(flipped, axis=self.dim)
        return np.flip(cum_result, axis=self.dim)


batch_size = 128
input_shape = (4000,)
dim = 1


def get_inputs():
    return [np.random.randn(batch_size, *input_shape).astype(np.float16)]


def get_init_inputs():
    return [dim]

