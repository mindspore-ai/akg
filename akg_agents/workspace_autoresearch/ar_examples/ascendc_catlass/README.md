# ascendc_catlass

Format: a CATLASS project directory plus sibling Python wrapper.
`--kernel` points at `catlass_op/`; its parent directory must contain
`kernel.py`.

Required layout:

```text
matmul/
  reference.py
  kernel.py
  catlass_op/
    CMakeLists.txt
    kernel/catlass_kernel.asc
    include/catlass_kernel.h
    src/catlass_torch.cpp
```

Run from `akg_agents/workspace_autoresearch` after setting
`defaults.dsl: ascendc_catlass`, `defaults.backend: ascend`,
`defaults.framework: torch`, and `catlass.root` or `CATLASS_ROOT`:

```text
/autoresearch --ref ar_examples/ascendc_catlass/matmul/reference.py --kernel ar_examples/ascendc_catlass/matmul/catlass_op --op-name catlass_matmul --devices 0
```

The adapter builds `catlass_op/build/libcatlass.so` before importing
`kernel.py`.
