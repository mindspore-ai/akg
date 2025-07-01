import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, dim: int = 1):
        self.dim = dim

    def __call__(self, x: np.ndarray) -> np.ndarray:
        max_val = np.max(x, axis=1, keepdims=True)
        shifted = x - max_val
        exp_vals = np.exp(shifted)
        sum_exp = np.sum(exp_vals, axis=1, keepdims=True)
        log_softmax = shifted - np.log(sum_exp)
        return log_softmax


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.randn(batch_size, dim).astype(np.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed

