# tilelang_npuir

Format: a single Python file exposing the function name expected by the
adapter:

```text
<op_name>_tilelang_npuir_<framework>(...)
```

With the command below, `op_name` is `vector_add` and `framework` is `torch`,
so the seed defines `vector_add_tilelang_npuir_torch(x, y)`.

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: tilelang_npuir`, `defaults.backend: ascend`,
`defaults.framework: torch`:

```text
/autoresearch --ref ar_examples/tilelang_npuir/reference.py --kernel ar_examples/tilelang_npuir/vector_add_kernel.py --op-name vector_add --devices 0
```
