# tilelang_ascend

Format: a single Python file with `class ModelNew(torch.nn.Module)`.
`ModelNew.__init__` compiles a TileLang AscendC kernel, and `forward` calls it.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: tilelang_ascend`, `defaults.backend: ascend`,
`defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/tilelang_ascend/reference.py --kernel ar_examples/tilelang_ascend/vector_add_kernel.py --op-name vector_add --devices 0
```
