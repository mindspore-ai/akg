import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        norm = np.sqrt(np.sum(np.square(x)))
        return x / norm


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = np.random.randn(batch_size, features, dim1, dim2).astype(np.float16)
    return [x]


def get_init_inputs():
    return []

