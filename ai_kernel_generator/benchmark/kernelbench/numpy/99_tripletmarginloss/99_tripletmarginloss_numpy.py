import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, margin=1.0):
        self.margin = margin

    def __call__(self, anchor, positive, negative) -> np.ndarray:
        distance_ap = np.linalg.norm(anchor - positive, axis=1)
        distance_an = np.linalg.norm(anchor - negative, axis=1)
        losses = np.maximum(distance_ap - distance_an + self.margin, 0.0)
        return np.mean(losses)


batch_size = 128
input_shape = (4096, )
dim = 1


def get_inputs():
    return [np.random.randn(batch_size, *input_shape).astype(np.float16), np.random.randn(batch_size,
                                                               *input_shape).astype(np.float16), np.random.randn(batch_size, *input_shape).astype(np.float16)]


def get_init_inputs():
    return [1.0]  # Default margin

