import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A, B) -> np.ndarray:
        return np.diag(A) @ B


M = 4096
N = 4096


def get_inputs():
    A = np.random.randn(N).astype(np.float16)
    B = np.random.randn(N, M).astype(np.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

