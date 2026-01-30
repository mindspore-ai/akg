# 转换建议

## 常见模式

1. **逐元素操作** → 直接使用 PyTorch 对应函数
2. **归约操作** → `torch.sum/max/min` 加 `dim` 参数
3. **Softmax** → `F.softmax(x, dim=-1)`
4. **矩阵乘法** → `torch.matmul(a, b)`

## Triton 转换问题

- ReLU: `torch.relu(x)` 或 `F.relu(x)`
- LayerNorm: `F.layer_norm(x, normalized_shape)`
- Attention: `F.scaled_dot_product_attention(q, k, v)`

## CUDA C 转换建议

1. **线程并行 → 向量化**：CUDA kernel 的线程并行直接用 PyTorch 向量化操作替代
2. **共享内存 → 无需处理**：PyTorch 自动管理内存
3. **同步操作 → 删除**：`__syncthreads()` 等同步操作无需保留
4. **load_inline → 删除**：删除 `torch.utils.cpp_extension.load_inline` 调用，使用原生 PyTorch

## 常见 CUDA C 转换示例

```python
# CUDA C 中的 ReLU kernel
# __global__ void relu_kernel(float* x, float* out, int n) {
#     int i = blockIdx.x * blockDim.x + threadIdx.x;
#     if (i < n) out[i] = max(x[i], 0.0f);
# }

# PyTorch 等价实现
def forward(self, x):
    return torch.relu(x)  # 或 torch.maximum(x, torch.zeros_like(x))
```
