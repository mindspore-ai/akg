import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self, num_features: int, num_groups: int):
        self.num_features = num_features
        self.num_groups = num_groups

    def __call__(self, x: np.ndarray) -> np.ndarray:
        eps = 1e-5
        batch_size, self.num_features, dim1, dim2 = x.shape
        groupsize = self.num_features // self.num_groups

        reshaped = x.reshape(batch_size, self.num_groups, groupsize, dim1, dim2)
        mean = np.mean(reshaped, axis=(2,3,4), keepdims=True)
        var = np.var(reshaped, axis=(2,3,4), keepdims=True)
        normalized = (reshaped - mean) / np.sqrt(var + eps)
        return normalized.reshape(x.shape)


batch_size = 16
features = 64
num_groups = 8
dim1 = 256
dim2 = 256


def get_inputs():
    x = np.random.randn(batch_size, features, dim1, dim2).astype(np.float16)
    return [x]


def get_init_inputs():
    return [features, num_groups]  # num_features

