# triton_cuda

Format: a single Python file with `class ModelNew(torch.nn.Module)`.
The `forward` method launches one or more `@triton.jit` kernels on CUDA.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: triton_cuda`, `defaults.backend: cuda`,
`defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/triton_cuda/reference.py --kernel ar_examples/triton_cuda/vector_add_kernel.py --op-name vector_add --devices 0
```
