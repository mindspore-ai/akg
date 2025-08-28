import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs Deep Normalization operation.
    Deep normalization is used in very deep transformer architectures to improve
    training stability by applying scaling factors that grow with network depth.
    """
    def __init__(self, alpha=0.87):
        super(Model, self).__init__()
        self.alpha = alpha

    def forward(self, tensor1, tensor2):
        """
        Perform Deep Normalization operation.
        
        Args:
            tensor1: First input tensor of shape (batch_size, seq_len, hidden_size)
            tensor2: Second input tensor of shape (batch_size, seq_len, hidden_size)
            
        Returns:
            Deep normalized output tensor
        """
        # Add residual connection
        added = tensor1 + tensor2
        
        # Apply layer normalization
        normalized_shape = added.shape[-1:]  # Normalize over the last dimension
        normalized = torch.nn.functional.layer_norm(added, normalized_shape)
        
        # Apply deep normalization scaling
        result = normalized * self.alpha
        
        return result


def get_inputs():
    """
    Generate random input tensors for testing with large model shapes.
    """
    # Batch size: 32, Sequence length: 1024, Hidden size: 4096
    batch_size, seq_len, hidden_size = 32, 1024, 4096
    tensor1 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    tensor2 = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.float32)
    return [tensor1, tensor2]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    alpha = 0.87
    return [alpha]