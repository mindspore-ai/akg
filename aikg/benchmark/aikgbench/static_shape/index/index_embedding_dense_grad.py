import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Model that performs Embedding Dense Gradient V2 operation.
    This operation computes gradients for embedding layers in a dense format, which is
    commonly used in training neural networks with embedding layers. The dense gradient
    format is more memory-efficient than sparse gradients for certain optimization algorithms.
    """

    def __init__(self, padding_idx=0):
        super(Model, self).__init__()
        self.padding_idx = padding_idx

    def forward(self, grad_output, indices, num_embeddings):
        """
        Perform Embedding Dense Gradient operation.

        Args:
            grad_output: Gradient output tensor of shape (batch_size, seq_len, embedding_dim) with float32 dtype
            indices: Indices tensor of shape (batch_size, seq_len) with int64 dtype
            num_embeddings: Number of embeddings (int)
            padding_idx: Padding index (int, default -1)

        Returns:
            Dense gradient tensor of shape (num_embeddings, embedding_dim)
        """
        # Initialize dense gradient tensor
        embedding_dim = grad_output.shape[-1]
        dense_grad = torch.zeros(num_embeddings, embedding_dim, dtype=torch.float32, device=grad_output.device)
        
        # Flatten indices and grad_output for easier processing
        flat_indices = indices.view(-1)
        flat_grad = grad_output.view(-1, embedding_dim)
        
        # Accumulate gradients for each unique index
        for i in range(len(flat_indices)):
            idx = flat_indices[i].item()
            if idx != self.padding_idx:
                dense_grad[idx] += flat_grad[i]
        
        return dense_grad


def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    # Batch size: 32
    # Sequence length: 1024
    # Embedding dimension: 4096
    batch_size, seq_len, embedding_dim = 32, 1024, 4096
    num_embeddings = 50257

    # Generate gradient output tensor
    grad_output = torch.randn(batch_size, seq_len, embedding_dim, dtype=torch.float32)
    
    # Generate indices tensor and inject padding positions
    indices = torch.randint(1, num_embeddings, (batch_size, seq_len), dtype=torch.int64)
    padding_ratio = 0.1
    padding_mask = torch.rand(batch_size, seq_len) < padding_ratio
    indices[padding_mask] = 0

    return [grad_output, indices, num_embeddings]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    padding_idx = 0
    return [padding_idx]
