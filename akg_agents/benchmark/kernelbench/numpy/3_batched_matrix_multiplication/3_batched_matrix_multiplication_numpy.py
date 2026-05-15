import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.matmul(A, B)


batch_size = 128
m = 128
k = 256
n = 512


def get_inputs():
    A = np.random.rand(batch_size, m, k).astype(np.float16)
    B = np.random.rand(batch_size, k, n).astype(np.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

