# AscendC Direct-Invoke Guide

AscendC tasks now use the direct-invoke project layout. Do not embed C++ source
inside Python string fields. The handoff is a normal Python wrapper plus a
normal AscendC/CMake project tree.

## Required Layout

```text
task_dir/
  kernel.py
  ascendc_op/
    CMakeLists.txt
    ...
```

`kernel.py` must define `ModelNew`. The verifier imports it with:

```python
from kernel import ModelNew
```

Then it constructs and calls the model as:

```python
impl_model = ModelNew(*init_params).to(device).eval()
impl_output = impl_model(*inputs)
```

The wrapper should return a `torch.Tensor` or a tuple/list of tensors. Keep
heavy work out of module import time; do lazy shared-library loading inside
`__init__` or `forward`.

## Wrapper Pattern

```python
from __future__ import annotations

from pathlib import Path

import torch
import torch_npu


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        project_dir = Path(__file__).with_name("ascendc_op")
        build_dir = project_dir / "build"
        # Load the extension produced by ascendc_op/CMakeLists.txt.
        # Replace the glob or file name with the project-specific module name.
        so_files = sorted(build_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError(f"no AscendC extension found under {build_dir}")
        torch.ops.load_library(str(so_files[0]))
        self._loaded = True

    def forward(self, *inputs):
        self._load()
        # Replace this with the pybind/torch.ops entrypoint exported by the
        # AscendC project.
        return torch.ops.my_namespace.my_op(*inputs)
```

## CMake Project Contract

The `ascendc_op/` directory is copied as a project tree. It must contain a
`CMakeLists.txt` that can be configured from an empty `build/` directory. The
adapter sets these variables when building:

```text
NPU_ARCH
ASCENDC_NPU_ARCH
ASCEND_HOME_PATH
TORCH_NPU_INCLUDE_DIR
TORCH_NPU_LIBRARY
Python_EXECUTABLE
Python3_EXECUTABLE
```

Prefer consuming `NPU_ARCH` from CMake. If the project contains a literal
`--npu-arch=...`, the adapter patches it to the detected direct-invoke value
before configuring.

## Rules For Generated Code

- Use `kernel.py` plus `ascendc_op/`; do not emit C++ source inside Python
  strings.
- Define `ModelNew` only once and make it the public entrypoint.
- Move tensors with `.to(device)`; do not hard-code `.npu()`.
- Do not run CMake, compile, or call `torch.ops.load_library` at module import
  time.
- Keep generated build output out of the project tree. `build/`, `output/`,
  `__pycache__/`, object files, and shared libraries are ignored when copying.
- Report missing shared libraries or missing torch operator symbols with clear
  exceptions from `_load()` or `forward()`.
