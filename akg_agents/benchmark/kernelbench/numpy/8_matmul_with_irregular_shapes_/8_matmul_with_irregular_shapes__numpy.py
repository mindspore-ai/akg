import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.matmul(A, B)


M = 8205
K = 2949
N = 5921


def get_inputs():
    A = np.random.rand(M, K).astype(np.float16)
    B = np.random.rand(K, N).astype(np.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

