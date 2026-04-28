import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Gated RMSNorm (PyTorch Reference Implementation).
    
    Used in Flash Linear Attention (FLA) models.
    Combines normalization with gating mechanism.
    
    Formula (norm_before_gate=True):
        x_normed = x * rsqrt(mean(x^2) + eps) * weight
        gate = sigmoid(z)
        out = x_normed * gate
    """

    def __init__(self, eps=1e-6, norm_before_gate=True, is_rms_norm=True):
        super(Model, self).__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.is_rms_norm = is_rms_norm

    def forward(self, x, weight, z=None):
        """
        Args:
            x: (batch, seq_len, hidden_dim) - input tensor
            weight: (hidden_dim,) - normalization weight
            z: (batch, seq_len, hidden_dim) - gating tensor (optional)
        
        Returns:
            out: (batch, seq_len, hidden_dim) - normalized and gated output
        """
        # RMSNorm
        if self.is_rms_norm:
            variance = x.pow(2).mean(dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(variance + self.eps) * weight
        else:
            # LayerNorm
            mean = x.mean(dim=-1, keepdim=True)
            variance = x.var(dim=-1, keepdim=True, unbiased=False)
            x_normed = (x - mean) * torch.rsqrt(variance + self.eps) * weight
        
        # Gating
        if z is not None:
            gate = torch.sigmoid(z)
            if self.norm_before_gate:
                out = x_normed * gate
            else:
                # Gate first, then normalize
                x_gated = x * gate
                variance = x_gated.pow(2).mean(dim=-1, keepdim=True)
                out = x_gated * torch.rsqrt(variance + self.eps) * weight
        else:
            out = x_normed
        
        return out


def get_inputs():
    batch = 32
    seq_len = 512
    hidden_dim = 4096
    
    x = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16)
    weight = torch.randn(hidden_dim, dtype=torch.float16)
    z = torch.randn(batch, seq_len, hidden_dim, dtype=torch.float16)
    
    return [x, weight, z]


def get_init_inputs():
    eps = 1e-6
    norm_before_gate = True
    is_rms_norm = True
    return [eps, norm_before_gate, is_rms_norm]
