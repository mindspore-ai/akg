import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A: np.ndarray, s: float) -> np.ndarray:
        return A * s


M = 16384
N = 4096


def get_inputs():
    A = np.random.randn(M, N).astype(np.float16)
    s = 3.14
    return [A, s]


def get_init_inputs():
    return []  # No special initialization inputs needed

