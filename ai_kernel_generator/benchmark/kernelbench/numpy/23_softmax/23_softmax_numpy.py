import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Numerical stability improvement by subtracting max value
        max_val = np.max(x, axis=1, keepdims=True)
        shifted = x - max_val
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
        return exp_vals / sum_exp


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.randn(batch_size, dim).astype(np.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed

