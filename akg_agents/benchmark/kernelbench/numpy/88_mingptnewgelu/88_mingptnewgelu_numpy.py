import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, x) -> np.ndarray:
        factor = np.sqrt(2.0 / np.pi)
        cubic = 0.044715 * np.power(x, 3.0)
        inner = factor * (x + cubic)
        tanh = np.tanh(inner)
        return 0.5 * x * (1.0 + tanh)


batch_size = 2000
dim = 2000


def get_inputs():
    return [np.random.rand(batch_size, dim).astype(np.float16)]


def get_init_inputs():
    return []

