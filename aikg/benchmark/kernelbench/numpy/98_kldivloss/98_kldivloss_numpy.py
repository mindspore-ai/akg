import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, predictions, targets) -> np.ndarray:
        log_p = np.log(predictions)
        loss_per_sample = np.sum(targets * (np.log(targets) - log_p), axis=1)
        loss = np.mean(loss_per_sample)
        return np.array([loss])


batch_size = 128
input_shape = (4096, )
dim = 1


def get_inputs():
    def softmax(x, axis=-1):
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)
    return [softmax(np.random.rand(batch_size, *input_shape).astype(np.float16)),
            softmax(np.random.rand(batch_size, *input_shape).astype(np.float16))]


def get_init_inputs():
    return []

