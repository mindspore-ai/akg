# cuda_c

Format: a single Python file with `class ModelNew(torch.nn.Module)`.
The model compiles CUDA C/C++ code through `torch.utils.cpp_extension`.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: cuda_c`, `defaults.backend: cuda`, `defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/cuda_c/reference.py --kernel ar_examples/cuda_c/vector_add_kernel.py --op-name vector_add --devices 0
```
