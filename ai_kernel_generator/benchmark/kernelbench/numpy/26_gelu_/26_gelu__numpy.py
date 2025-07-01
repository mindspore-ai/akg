import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        sqrt_2_over_pi = np.sqrt(2 / np.pi)
        inner = x + 0.044715 * (x ** 3)
        tanh_term = np.tanh(sqrt_2_over_pi * inner)
        return 0.5 * x * (1 + tanh_term)


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.randn(batch_size, dim).astype(np.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed

