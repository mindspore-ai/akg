import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        l2_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)
        return x / l2_norm


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.rand(batch_size, dim).astype(np.float16)
    return [x]


def get_init_inputs():
    return []

