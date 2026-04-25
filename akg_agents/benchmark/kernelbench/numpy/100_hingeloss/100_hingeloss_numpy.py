import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, predictions, targets) -> np.ndarray:
        loss = 1 - predictions * targets
        loss_clipped = np.clip(loss, a_min=0, a_max=None)
        mean_loss = np.mean(loss_clipped)
        return np.array([mean_loss])


batch_size = 128
input_shape = (1,)
dim = 1


def get_inputs():
    return [np.random.rand(batch_size, *input_shape).astype(np.float16), np.random.randint(0, 2, (batch_size, 1)).astype(float) * 2 - 1]


def get_init_inputs():
    return []

