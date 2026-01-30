import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, x * self.negative_slope)


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.rand(batch_size, dim).astype(np.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed

