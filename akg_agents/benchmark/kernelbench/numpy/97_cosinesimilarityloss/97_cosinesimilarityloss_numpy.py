import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, predictions, targets) -> np.ndarray:
        dot_product = np.sum(predictions * targets, axis=1)
        norm_p = np.linalg.norm(predictions, axis=1)
        norm_t = np.linalg.norm(targets, axis=1)
        cosine_sim = dot_product / (norm_p * norm_t)
        loss = np.mean(1 - cosine_sim)
        return np.array([loss])


batch_size = 128
input_shape = (4096, )
dim = 1


def get_inputs():
    return [np.random.rand(batch_size, *input_shape).astype(np.float16), np.random.rand(batch_size, *input_shape).astype(np.float16)]


def get_init_inputs():
    return []

