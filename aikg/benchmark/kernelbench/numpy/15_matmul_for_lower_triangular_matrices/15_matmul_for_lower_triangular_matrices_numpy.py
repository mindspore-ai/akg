import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A, B) -> np.ndarray:
        product = np.matmul(A, B)
        return np.tril(product)


M = 4096


def get_inputs():
    A = np.random.randn(M, M).astype(np.float16)
    B = np.random.randn(M, M).astype(np.float16)
    A = np.tril(A)
    B = np.tril(B)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

