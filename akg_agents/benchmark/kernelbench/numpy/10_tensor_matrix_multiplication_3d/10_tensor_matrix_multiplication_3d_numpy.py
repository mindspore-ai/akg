import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A, B) -> np.ndarray:
        return np.matmul(A, B)


N = 16
M = 1024
K = 2048
L = 768


def get_inputs():
    A = np.random.rand(N, M, K).astype(np.float16)
    B = np.random.rand(K, L).astype(np.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

