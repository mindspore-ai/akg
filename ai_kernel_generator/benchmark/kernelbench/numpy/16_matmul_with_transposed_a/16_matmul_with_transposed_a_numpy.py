import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.matmul(A.T, B)


M = 1024
K = 4096
N = 2048


def get_inputs():
    A = np.random.randn(K, M).astype(np.float16)
    B = np.random.randn(K, N).astype(np.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

