import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, num_features: int):
        self.num_features = num_features

    def __call__(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-5
        mean = np.mean(x, axis=(2, 3), keepdims=True)
        var = np.var(x, axis=(2, 3), keepdims=True)
        normalized = (x - mean) / np.sqrt(var + eps)
        return normalized


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = np.random.randn(batch_size, features, dim1, dim2).astype(np.float16)
    return [x]


def get_init_inputs():
    return [features]

