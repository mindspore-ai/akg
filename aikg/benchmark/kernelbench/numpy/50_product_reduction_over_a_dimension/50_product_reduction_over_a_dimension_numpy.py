import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, dim: int):
        self.dim = dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.prod(x, axis=1)


batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1


def get_inputs():
    x = np.random.randn(batch_size, dim1, dim2).astype(np.float16)
    return [x]


def get_init_inputs():
    return [reduction_dim]

