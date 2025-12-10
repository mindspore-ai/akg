# 转换建议

## 常见模式

1. **逐元素操作** → 直接使用 PyTorch 对应函数
2. **归约操作** → `torch.sum/max/min` 加 `dim` 参数
3. **Softmax** → `F.softmax(x, dim=-1)`
4. **矩阵乘法** → `torch.matmul(a, b)`

## 常见问题

- ReLU: `torch.relu(x)` 或 `F.relu(x)`
- LayerNorm: `F.layer_norm(x, normalized_shape)`
- Attention: `F.scaled_dot_product_attention(q, k, v)`
