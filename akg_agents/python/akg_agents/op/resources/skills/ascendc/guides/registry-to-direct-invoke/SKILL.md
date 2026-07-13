---
name: ascendc-registry-to-direct-invoke
description: "把注册式 AscendC 算子迁移为 direct-invoke 工程的保真原则：保持 kernel 算法和 tiling 公式，仅替换注册框架胶水、入口 ABI、host launch 与 PyTorch extension。"
category: guide
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  operator_patterns: "all"
---

# 注册式工程迁移到 Direct-Invoke

当已有 AscendC 自定义算子或 registry-invoke 工程需要迁移到 `kernel.py + ascendc_op/` 布局时使用本 skill。

## 1. 迁移原则

只替换注册框架胶水。除非任务明确要求算法重写，否则保持 kernel 算法、tiling 公式和 dtype/shape 分支不变。

## 2. 保持 Kernel 逻辑

迁移时不要改动：

- kernel class 的布局、成员变量、成员函数。
- `CopyIn`、`Compute`、`CopyOut`、`Process` 的顺序。
- compute 和 data movement 使用的 helper 函数。
- loop 结构、branch 结构和算术表达式顺序。
- dtype-specific path 与 tiling-key-specific path。

允许的机械修改：

- 将跨目录 include 改成本地 include，或补一个小型本地依赖头。
- 为避免符号冲突添加 namespace。
- 将框架宏生成的 tiling 类型替换成等价 POD struct。
- 调整 kernel entry ABI，使其匹配 direct launch。

## 3. 保持 Tiling 公式

以下公式和分支应逐项保留：

- core 数、tile 长度、基础 shape、对齐、tail 计算。
- aligned 与 non-aligned 分支。
- dtype、quant、grad、full-load、split-D 等变体分支。
- 分子/分母顺序、ceil/floor 行为。

不要为了减少代码合并看似相近的 tiling 分支。tiling drift 很可能编译通过，但静默写错结果。

## 4. 替换工程胶水

direct-invoke 需要的胶水包括：

- kernel entry：显式参数的 direct launcher。
- tiling data：host/kernel 共用的普通 POD struct。
- host launch：计算 tiling，分配输出，传入 stream，启动 kernel。
- PyTorch extension：注册 `torch.ops.npu.<op>`，实现 Meta 函数。
- `kernel.py`：懒加载 `.so`，在 `ModelNew.forward()` 中调用注册算子。

以下 registry 宏不应保留在 direct-invoke seed 中：

- `IMPL_OP_OPTILING`
- `REGISTER_TILING_DATA_CLASS`
- 依赖 registry glue 的运行时 `GET_TILING_DATA` dispatch
- direct 工程中不可用的框架专用 `OP_LOG*` 宏

必要日志可以改成 host 侧异常、调试期 `printf` 或 Python 侧异常；最终提交前移除重型调试输出。

## 5. 依赖闭包

当前算子的源文件建议整文件复制。外部公共 helper 只复制最小符号闭包：

- SDK include 保持 SDK include。
- 小型常量和 traits 可以本地化。
- helper 函数必须连同直接依赖一起复制。
- 不要把大段无关 common header tree 拖入 `ascendc_op/`。

迁移后应检查：

```bash
rg '#include "\\.\\.' ascendc_op
```

除非相对 include 仍在复制后的工程树内部，否则结果应为空。

## 6. Host/Device 编译 Pass 保护

AscendC `.asc` 或 `-xasc` 编译可能对同一源文件执行 host pass 和 device pass。若实现头使用 device-only MicroAPI 或 RegBase 类型，应保护实现体：

```cpp
#if !defined(__NPU_HOST__)
// device-only kernel implementation
#endif
```

纯 POD tiling header 同时被 host 和 device 使用，不应被上述宏包住。

## 7. 验证清单

- host declaration、generated launcher、kernel entry 的 ABI 完全一致。
- tensor 指针按照 launcher 约定一致传为 `GM_ADDR` 或 `uint8_t*`。
- tiling struct 的字段名、类型、顺序、大小在 host/kernel 两侧一致。
- PyTorch Meta 实现返回正确 shape 和 dtype。
- `kernel.py` 不在 import 阶段编译、改源码或选设备。
- `ascendc_op/` 与同级 `kernel.py` 构成完整任务产物。
