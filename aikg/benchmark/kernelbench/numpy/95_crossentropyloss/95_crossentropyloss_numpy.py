import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, predictions, targets) -> np.ndarray:
        max_vals = np.max(predictions, axis=1, keepdims=True)
        max_vals = np.max(predictions, axis=1, keepdims=True)
        shifted = predictions - max_vals
        exp_shifted = np.exp(shifted)
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        log_softmax = shifted - np.log(sum_exp)

        batch_indices = np.arange(predictions.shape[0])
        selected = log_softmax[batch_indices, targets]

        loss = -np.mean(selected)
        return np.array([loss])


batch_size = 4096
num_classes = 10
input_shape = (num_classes, )  # Output for each class
dim = 1


def get_inputs():
    return [np.random.rand(batch_size, *input_shape).astype(np.float16), np.random.randint(0, num_classes, (batch_size,))]


def get_init_inputs():
    return []

