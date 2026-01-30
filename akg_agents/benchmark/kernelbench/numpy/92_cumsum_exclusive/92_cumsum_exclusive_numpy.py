import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, dim):
        self.dim = dim

    def __call__(self, x) -> np.ndarray:
        zeros = np.zeros_like(x[:, 0:1])
        concatenated = np.concatenate([zeros, x], axis=1)
        sliced = concatenated[:, :-1]
        return np.cumsum(sliced, axis=1)


batch_size = 128
input_shape = (4000,)
dim = 1


def get_inputs():
    return [np.random.rand(batch_size, *input_shape).astype(np.float16)]


def get_init_inputs():
    return [dim]

