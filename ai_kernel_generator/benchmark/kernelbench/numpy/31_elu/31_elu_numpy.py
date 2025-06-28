import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))


batch_size = 16
dim = 16384


def get_inputs():
    x = np.random.randn(batch_size, dim).astype(np.float16)
    return [x]


def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization

