import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, predictions, targets) -> np.ndarray:
        diff = predictions - targets
        squared_diff = np.square(diff)
        return np.mean(squared_diff)


batch_size = 128
input_shape = (4096, )
dim = 1


def get_inputs():
    return [np.random.randn(batch_size, *input_shape).astype(np.float16), np.random.randn(batch_size, *input_shape).astype(np.float16)]


def get_init_inputs():
    return []

