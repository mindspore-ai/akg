import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A, B) -> np.ndarray:
        return np.einsum('bijl,lk->bijk', A, B)


b = 16
i = 256
j = 512
l = 256
k = 768


def get_inputs():
    A = np.random.randn(b, i, j, l).astype(np.float16)
    B = np.random.randn(l, k).astype(np.float16)
    return [A, B]


def get_init_inputs():
    return []  # No special initialization inputs needed

