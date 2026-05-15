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
        
        # Apply layer normalization (normalize over the last dimension)
        normalized_shape = added.shape[-1:]
        normalized = torch.nn.functional.layer_norm(added, normalized_shape)
        
        # Apply deep normalization scaling
        result = normalized * self.alpha
        
        return result

def get_inputs_dyn_list():
    """
    Generate multiple sets of random input tensors for testing with different shapes.
    """
    # Case 1: Small tensor size (16, 512, 2048) (smaller than static)
    batch_size1, seq_len1, hidden_size1 = 16, 512, 2048
    tensor1_1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    tensor2_1 = torch.randn(batch_size1, seq_len1, hidden_size1, dtype=torch.float32)
    
    # Case 2: Medium tensor size (24, 768, 3072) (non-aligned batch, medium hidden)
    batch_size2, seq_len2, hidden_size2 = 24, 768, 3072
    tensor1_2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    tensor2_2 = torch.randn(batch_size2, seq_len2, hidden_size2, dtype=torch.float32)
    
    # Case 3: Large tensor size (32, 1024, 4096) (aligned, same as static)
    batch_size3, seq_len3, hidden_size3 = 32, 1024, 4096
    tensor1_3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    tensor2_3 = torch.randn(batch_size3, seq_len3, hidden_size3, dtype=torch.float32)
    
    # Case 4: Very large tensor size (48, 1536, 6144) (non-aligned batch, larger than static)
    batch_size4, seq_len4, hidden_size4 = 48, 1536, 6144
    tensor1_4 = torch.randn(batch_size4, seq_len4, hidden_size4, dtype=torch.float32)
    tensor2_4 = torch.randn(batch_size4, seq_len4, hidden_size4, dtype=torch.float32)
    
    return [
        [tensor1_1, tensor2_1],
        [tensor1_2, tensor2_2],
        [tensor1_3, tensor2_3],
        [tensor1_4, tensor2_4]
    ]

def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    alpha = 0.87
    return [alpha]