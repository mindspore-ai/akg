# tilelang_cuda

Format: a single Python file with `class ModelNew(torch.nn.Module)`.
`ModelNew.__init__` compiles the TileLang kernel, and `forward` calls it.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: tilelang_cuda`, `defaults.backend: cuda`,
`defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/tilelang_cuda/reference.py --kernel ar_examples/tilelang_cuda/vector_add_kernel.py --op-name vector_add --devices 0
```
