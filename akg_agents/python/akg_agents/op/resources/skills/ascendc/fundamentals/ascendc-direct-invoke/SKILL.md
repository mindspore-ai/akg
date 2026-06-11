---
name: ascendc-direct-invoke
description: "AscendC direct-invoke 工程契约：WA 使用 kernel.py + ascendc_op/，ModelNew 调用 torch.ops.npu.*，adapter 负责复制工程、CMake 构建和 npu-arch patch。适用于 dsl=ascendc 的 autoresearch 任务。"
---

# AscendC Direct-Invoke

This skill is adapted for AKG autoresearch from the CANNBot direct-invoke
template. In WA, the kernel handoff is not the old three-string AscendC
protocol and not a registry-invoke package. It is a Python wrapper plus a
normal CMake AscendC project:

```text
task_dir/
  kernel.py
  ascendc_op/
    CMakeLists.txt
    op_kernel/
    op_host/
    op_extension/
    scripts/
```

## Hard Contract

- `kernel.py` defines exactly one public entry class named `ModelNew`.
- `ModelNew.forward()` calls the compiled extension through `torch.ops.npu.<op>(...)` or the namespace registered by the project.
- `ascendc_op/` contains the CMake project; do not embed C++ source in Python strings.
- Do not call CMake or compile from `kernel.py`. AKG's AscendC adapter copies `ascendc_op/`, patches `--npu-arch=...`, configures CMake, builds, then imports `ModelNew`.
- Do not use `.npu()` in wrapper code. Inputs are already moved by the verifier; use ordinary PyTorch tensors on the selected device.
- Load the `.so` lazily inside `ModelNew._load()` or `forward()`, not at module import time.

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
        build_dir = Path(__file__).with_name("ascendc_op") / "build"
        so_files = sorted(build_dir.rglob("*.so"))
        if not so_files:
            raise RuntimeError(f"no AscendC extension found under {build_dir}")
        torch.ops.load_library(str(so_files[0]))
        self._loaded = True

    def forward(self, x, y):
        self._load()
        return torch.ops.npu.my_op(x, y)
```

Keep `_load()` deterministic. If multiple shared libraries exist, prefer a
stable name match such as `libmy_op_ops.so` before falling back to sorted order.

## Project Pattern

The direct-invoke project usually has two targets:

- an optional executable for standalone kernel launch and local debugging
- a shared library registered through `TORCH_LIBRARY_FRAGMENT` /
  `TORCH_LIBRARY_IMPL`

Common files:

```text
ascendc_op/
  CMakeLists.txt
  op_kernel/<op>_kernel.asc
  op_kernel/<op>_tiling.h
  op_host/<op>.asc
  op_extension/<op>_torch.cpp
  op_extension/register.cpp
  op_extension/ops.h
```

The PyTorch extension side follows this shape:

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("my_op(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("my_op", TORCH_FN(ascend_kernel::my_op_torch));
}
```

The host launch code should:

- allocate output tensors with PyTorch on the same device as inputs
- compute tiling from tensor shape and dtype
- copy tiling data to device memory or a device tensor
- get the current NPU stream via `c10_npu::getCurrentNPUStream()`
- call the generated kernel launcher with ABI matching the kernel entry

## CMake Requirements

AKG passes these variables to CMake:

```text
NPU_ARCH
ASCENDC_NPU_ARCH
ASCEND_HOME_PATH
Python_EXECUTABLE
Python3_EXECUTABLE
```

If `CMakeLists.txt` contains a literal `--npu-arch=dav-2201` or
`--npu-arch=dav-3510`, the adapter patches it for the current arch. Prefer
using `${NPU_ARCH}` directly when writing new projects.

The project must build a `.so` somewhere under `ascendc_op/build/`. The adapter
searches recursively.

## Vector Template Route

For elementwise and simple reduction operators, use the CANNBot `add_custom`
pattern as the mental model:

- `op_kernel/<op>_tiling.h`: tiling constants and POD struct shared by host and kernel
- `op_kernel/<op>_kernel.asc`: `KernelXxx` class with `Init`, `CopyIn`, `Compute`, `CopyOut`, `Process`
- `op_extension/<op>_torch.cpp`: PyTorch bridge, tiling computation, stream selection, kernel launch
- `register.cpp`: `torch.ops.npu.<op>` registration and Meta implementation

When adapting it, globally rename the operator and then modify only marked
semantic points: input/output count, tiling fields, compute expression, output
shape, dtype checks, and CMake target names.

## Do Not Generate

- `host_tiling_src = """..."""`
- `kernel_src = """..."""`
- `python_bind_src = """..."""`
- `run.sh`-only workflows inside WA
- import-time compilation
- hard-coded local device IDs

WA invokes the formal `KernelVerifier` chain. The seed must be a reusable
project, not a one-off script.
