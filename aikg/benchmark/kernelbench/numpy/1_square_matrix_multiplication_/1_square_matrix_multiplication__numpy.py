import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.matmul(A, B)


N = 2048


def get_inputs():
    A = np.random.randn(N, N).astype(np.float16)
    B = np.random.randn(N, N).astype(np.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

