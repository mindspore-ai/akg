---
name: pypto-case-matvec
description: "矩阵-向量乘法：K > 65535 时用 elementwise mul + sum 替代 matmul"
category: case
version: "1.0.0"
metadata:
  backend: ascend
  dsl: pypto
  operator_patterns: "matrix_vector,matvec,large_k"
---

# Matrix-Vector Multiplication (K > 65535)

A: (256, 131072), B: (131072, 1) -> C: (256, 1)

K=131072 超过 `pypto.matmul` 限制（最后一维 <= 65535），用 `sum(a * b_row, dim=1)` 替代。

```python
def create_matvec_sum_kernel(a_shape, b_shape):
    out_shape = (a_shape[0], 1)

    @pypto.frontend.jit(...)
    def matvec_sum_kernel(
            a: pypto.Tensor(a_shape, pypto.DT_FP32),
            b_row: pypto.Tensor(b_shape, pypto.DT_FP32),
    ) -> pypto.Tensor(out_shape, pypto.DT_FP32):
        output = pypto.tensor(list(out_shape), pypto.DT_FP32)
        pypto.set_vec_tile_shapes(1, 8192)
        output[:] = pypto.sum(a * b_row, dim=1, keepdim=True)
        return output
    return matvec_sum_kernel

class ModelNew(torch.nn.Module):
    def forward(self, A, B):
        assert A.dim() == 2
        assert tuple(A.shape) == (256, 131072)
        assert B.dim() == 2
        assert tuple(B.shape) == (131072, 1)
        A = A.contiguous()
        # B: (K, 1) -> (1, K) 用于广播乘法
        B_row = B.contiguous().reshape(1, -1)
        return create_matvec_sum_kernel(tuple(A.shape), tuple(B_row.shape))(A, B_row)
```

关键点：forward 中 `B.reshape(1, -1)` 将列向量转为行向量，使 `a * b_row` 可广播。
