import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A, B) -> np.ndarray:
        product = np.matmul(A, B)
        return np.triu(product)


N = 4096


def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = np.triu(np.random.randn(N, N).astype(np.float16))
    B = np.triu(np.random.randn(N, N).astype(np.float16))
    return [A, B]


def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []

