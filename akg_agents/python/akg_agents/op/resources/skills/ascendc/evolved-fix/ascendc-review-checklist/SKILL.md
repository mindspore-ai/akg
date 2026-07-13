---
name: ascendc-review-checklist
description: "AscendC 算子修改前后的快速审查清单：direct-invoke wrapper、CMake/注册、launch ABI、tiling、DataCopy、同步、dtype/shape 覆盖和不可静默降级。"
category: fix
version: "1.0.0"
metadata:
  backend: ascend
  dsl: ascendc
  hardware: "Atlas A2, Atlas A3, Atlas A5"
  case_type: review
---

# AscendC 修改审查清单

在提交前、批跑前，或代码看起来合理但验证失败时使用本 skill。它用于快速排除常见假设错误，不能替代真实 verify。

## 1. Wrapper 与构建

- `kernel.py` 只定义一个公开 `ModelNew`。
- `ModelNew` 懒加载构建出的 `.so`，没有 import 阶段编译。
- 多个 `.so` 存在时，加载目标库的规则稳定。
- `torch.ops.npu.<op>` namespace 与 C++ 注册一致。
- Meta 函数返回精确输出 shape 和 dtype。
- CMake 在 `ascendc_op/build` 下生成 shared library。
- `--npu-arch` 由 CMake 变量或适配器 patch 控制，没有过期硬编码。

## 2. Launch ABI

host declaration、launcher、kernel entry 需要一致：

- 参数数量。
- 参数顺序。
- 指针类型约定。
- tiling pointer/tensor 参数。
- workspace pointer。
- stream 参数。
- `blockDim`。

修改 kernel entry 参数时，必须同步修改 host bridge 和 Python 调用路径。

## 3. Tiling 与内存

- tiling 字段顺序和类型在 host/kernel 两侧一致。
- `SetGlobalBuffer` span 等于当前 core 实际可访问元素数。
- `GetBlockIdx()` 计算出的 offset 不会超过 total。
- tail 路径使用真实 tail length 和有效 vector mask。
- `DataCopy` 长度单位正确。
- 非对齐搬运使用 `DataCopyPad` 或专门 tail 路径。

## 4. Queue 与同步

- 每个 `AllocTensor` 都被释放。
- 每个 `EnQue` 都有匹配 `DeQue`。
- 没有分支在 allocation 后跳过 cleanup。
- 跨核等待在所有参与 core 上都有必达 setter。
- barrier 对应真实数据依赖，不用作不明原因的兜底。

## 5. 数值与覆盖

- reduction、exp/log/div/sqrt 等敏感路径按参考精度使用 fp32 中间结果。
- float-to-half、half-to-float 的 Cast round mode 显式。
- dtype 分支覆盖任务要求的可见范围。
- shape-specific path 不能被静默合并，除非任务明确允许。
- tolerance 修改不能用于掩盖实现错误。

## 6. 批跑安全

- 单个 op 的修复不应修改共享 scaffold，导致其他 op 失效。
- 若简化移除了 dtype、layout 或 shape 路径，需要在报告中写明 deliberate exclusion。
- 优先做小范围局部修改，并附上新的验证证据。

## 7. 性能优化专项检查

性能改动通过 correctness 不代表值得保留。提交前额外检查：

- 是否只改变了一个明确性能假设，例如 tile、同步、cast、合批、索引外提中的一个。
- 是否同时查看总指标和逐样本指标，避免单个样本收益掩盖多数样本回退。
- 窄路径条件是否是语义条件：dtype、rank、contiguous、broadcast、reduce axis、numel、特殊值模式，而不是无解释的完整 shape。
- 库实现旁路是否只覆盖明确退化模式，并且主 AscendC 路径仍覆盖普通输入。
- 是否因为快路径遗漏了非 contiguous、反向 broadcast、空 tensor、tail、非 32B 对齐或 dtype 变体。
- tile、BUFFER_NUM、rowsPerTile 改动是否重新计算 UB 账本。
- 删除同步是否有数据依赖证明；不要把 barrier 删成偶然正确。
- 删除 dead code 是否同步清理 CMake、extern declaration、register、Meta 和 launcher。
- 如果使用 in-place 或 alias 输出，必须确认调用语义允许覆盖输入。

逐样本性能报告建议记录：

```text
dominant slow samples:
shared semantic pattern:
changed path:
total metric before/after:
worst regression sample:
coverage preserved:
rollback condition:
```

## 8. 提交前最小检查

```text
build/load:
correctness shapes:
tail/non-aligned shapes:
dtype variants:
known exclusions:
profiling status:
```

上述字段为空时，不要把性能结论当成最终结论。
