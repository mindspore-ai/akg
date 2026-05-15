import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, predictions, targets) -> np.ndarray:
        beta = 1.0
        diff = predictions - targets
        abs_diff = np.abs(diff)
        loss = np.where(abs_diff < beta, 0.5 * (diff ** 2) / beta, abs_diff - 0.5 * beta)
        return np.mean(loss)


batch_size = 128
input_shape = (4096, )
dim = 1


def get_inputs():
    return [np.random.rand(batch_size, *input_shape).astype(np.float16), np.random.rand(batch_size, *input_shape).astype(np.float16)]


def get_init_inputs():
    return []

