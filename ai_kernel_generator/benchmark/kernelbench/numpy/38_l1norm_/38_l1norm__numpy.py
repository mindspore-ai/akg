import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sum_abs = np.sum(np.abs(x), axis=1, keepdims=True)
        return x / sum_abs


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.randn(batch_size, dim).astype(np.float16)
    return [x]


def get_init_inputs():
    return []

