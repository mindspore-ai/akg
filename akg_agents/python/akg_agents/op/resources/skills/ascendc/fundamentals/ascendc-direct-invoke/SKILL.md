---
name: ascendc-direct-invoke
description: "AscendC direct-invoke 工程契约：任务目录使用 kernel.py + ascendc_op/，ModelNew 调用 torch.ops.npu.*，适配器负责复制工程、CMake 构建和 npu-arch patch。适用于 dsl=ascendc 的算子生成与验证任务。"
category: fundamental
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# AscendC Direct-Invoke 工程契约

`dsl=ascendc` 的任务目录不是旧的三段字符串协议，也不是完整注册式自定义算子工程。标准交付形态是一个 Python wrapper 加一个可由 CMake 构建的 AscendC 工程：

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

## 1. 硬性约定

- `kernel.py` 只暴露一个公开入口类：`ModelNew`。
- `ModelNew.forward()` 通过 `torch.ops.npu.<op>(...)` 或工程注册的 namespace 调用编译后的扩展。
- `ascendc_op/` 保存 CMake 工程；不要在 Python 字符串里内嵌 C++/AscendC 源码。
- `kernel.py` 不负责调用 CMake，也不在 import 阶段编译。
- wrapper 代码不要调用 `.npu()`；验证器会把输入移动到目标设备。
- `.so` 在 `ModelNew._load()` 或 `forward()` 中懒加载，避免 import 阶段副作用。

## 2. Python Wrapper 模板

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

若 `build/` 下可能存在多个 `.so`，优先按稳定文件名筛选，例如 `libmy_op_ops.so`；最后再使用排序后的第一个结果。

## 3. CMake 工程结构

常见工程包含两个部分：

- 可选的 standalone 可执行文件，用于本地 kernel launch 和调试。
- PyTorch shared library，通过 `TORCH_LIBRARY_FRAGMENT` / `TORCH_LIBRARY_IMPL` 注册算子。

推荐文件布局：

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

PyTorch 扩展注册示例：

```cpp
TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("my_op(Tensor x, Tensor y) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu, PrivateUse1, m) {
    m.impl("my_op", TORCH_FN(ascend_kernel::my_op_torch));
}
```

## 4. Host Launch 职责

host 侧桥接代码需要完成：

- 在输入所在设备上分配输出 tensor。
- 从 shape、dtype、stride 推导 tiling。
- 将 tiling 数据复制到设备侧可读的内存或 tensor。
- 通过 `c10_npu::getCurrentNPUStream()` 获取当前 NPU stream。
- 按 kernel entry 的 ABI 顺序调用 launcher。

Meta 函数必须返回精确的输出 shape 和 dtype。错误的 Meta 会导致验证器读取错误 shape，进而把正确 kernel 判断为失败。

## 5. CMake 约定

适配器会向 CMake 传入：

```text
NPU_ARCH
ASCENDC_NPU_ARCH
ASCEND_HOME_PATH
Python_EXECUTABLE
Python3_EXECUTABLE
```

新工程优先使用 `${NPU_ARCH}`。若旧工程中存在 `--npu-arch=dav-2201` 或 `--npu-arch=dav-3510` 之类硬编码，构建适配层会尝试 patch，但不要在新代码里继续写死。

工程必须在 `ascendc_op/build/` 下生成可加载的 `.so`；搜索是递归的，但文件名稳定会降低误加载风险。

## 6. Vector 类算子改造路径

elementwise、broadcast、简单 reduction 可以采用统一的 Vector 模板：

- `op_kernel/<op>_tiling.h`：host/kernel 共用的 tiling struct。
- `op_kernel/<op>_kernel.asc`：`KernelXxx` 类，包含 `Init`、`CopyIn`、`Compute`、`CopyOut`、`Process`。
- `op_extension/<op>_torch.cpp`：PyTorch 桥接、tiling 计算、stream 选择、kernel launch。
- `register.cpp`：`torch.ops.npu.<op>` 注册和 Meta 实现。

迁移已有模板时，先全局替换算子名，再只改语义点：输入/输出数量、tiling 字段、计算表达式、输出 shape、dtype 检查、CMake target 名称。

## 7. 禁止生成的形态

不要生成：

- `host_tiling_src = """..."""` 这类 Python 内嵌源码。
- `kernel_src = """..."""` 这类运行时拼接源码。
- 只依赖 `run.sh` 的一次性工作流。
- import 阶段编译、import 阶段选设备。
- 硬编码本地 device id。

任务产物应是可重复构建、可由验证链路接管的工程，而不是单次本地脚本。
