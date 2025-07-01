import numpy as np

class Model:
    """
    NumPy实现的模型类
    """

    def __init__(self):
        pass

    def __call__(self, A, B) -> np.ndarray:
        return np.matmul(A, B)


N = 4096


def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = np.random.randn(N, N).astype(np.float16)
    A = (A + A.T) / 2  # Ensure symmetry
    B = np.random.randn(N, N).astype(np.float16)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]


def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []

