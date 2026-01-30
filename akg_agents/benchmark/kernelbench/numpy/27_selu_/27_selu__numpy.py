import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, x: np.ndarray) -> np.ndarray:
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        condition = x > 0
        return scale * np.where(condition, x, alpha * (np.exp(x) - 1))


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.rand(batch_size, dim).astype(np.float16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed

