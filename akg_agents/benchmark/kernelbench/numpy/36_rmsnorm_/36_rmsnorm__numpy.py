import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.num_features = num_features
        self.eps = eps

    def __call__(self, x: np.ndarray) -> np.ndarray:
        squared = np.square(x)
        mean_squared = np.mean(squared, axis=1, keepdims=True)
        rms = np.sqrt(mean_squared + self.eps)
        return x / rms


batch_size = 16
features = 64
dim1 = 256
dim2 = 256


def get_inputs():
    x = np.random.rand(batch_size, features, dim1, dim2).astype(np.float16)
    return [x]


def get_init_inputs():
    return [features]

