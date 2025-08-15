import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Simple model that performs DynamicQuantUpdateScatter operation.
    """

    def __init__(self, reduce="add", axis=0):
        super(Model, self).__init__()
        self.reduce = reduce
        self.axis = axis

    def forward(self, var, var_scale, indices, updates, smooth_scales):
        """
        Perform DynamicQuantUpdateScatter operation.

        Args:
            var: Quantized variable tensor (int8)
            var_scale: Variable scale tensor (float32)
            indices: Indices tensor
            updates: Updates tensor (float16/bfloat16)
            smooth_scales: Smooth scales tensor (float16/bfloat16)

        Returns:
            Tuple of (y, var, var_scale) tensors
        """
        # Dequantize var using var_scale
        var_dequantized = var.float() * var_scale

        # Apply smooth scales to updates
        scaled_updates = updates * smooth_scales

        # Create a copy of var_dequantized for scatter operation
        output = var_dequantized.clone()

        # Ensure data type compatibility
        output = output.to(torch.float32)
        scaled_updates = scaled_updates.to(torch.float32)

        # Perform scatter operation based on reduce mode
        if self.reduce == "add":
            output.scatter_add_(self.axis, indices, scaled_updates)
        else:
            output.scatter_(self.axis, indices, scaled_updates)

        # Re-quantize the result
        # Find the scale for the updated tensor
        max_val = torch.abs(output).max()
        new_scale = max_val / 127.0  # Assuming int8 quantization

        # Quantize to int8
        y = torch.clamp(torch.round(output / new_scale), -
                        128, 127).to(torch.int8)

        return y, output, new_scale.unsqueeze(0)


def get_inputs():
    """
    Generate random input tensors for testing.
    """
    # Create tensors with appropriate shapes and types
    var_shape = [4, 4]
    indices_shape = [2, 4]

    var = torch.randint(-128, 127, var_shape, dtype=torch.int8)
    var_scale = torch.randn(1, dtype=torch.float32)
    indices = torch.randint(0, 4, indices_shape, dtype=torch.int64)  # 改为 int64
    updates = torch.randn(indices_shape, dtype=torch.float16)
    smooth_scales = torch.randn(indices_shape, dtype=torch.float16)

    return [var, var_scale, indices, updates, smooth_scales]


def get_init_inputs():
    """
    Return initialization parameters for the model.
    """
    return ["add", 0]  # reduce="add", axis=0
