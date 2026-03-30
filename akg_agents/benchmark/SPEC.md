# benchmark/ — 评测规范

## 职责

存放算子/内核生成的评测集，用于衡量生成质量和覆盖率。

## 目录结构

| 子目录 | 说明 |
|--------|------|
| `kernelbench/` | 适配后的 KernelBench 任务（MindSpore / NumPy 实现） |
| `sol-execbench/` | SOL-ExecBench 评测集下载脚本和说明文档 |
| `akg_kernels_bench/` | AKG 高频内核场景集（静态/动态 shape、融合、attention 等，含 Triton Ascend 子树） |
| `akg_kernels_bench_lite/` | 精简 Ascend NPU 测例（分 t1/t2/t3 难度层级） |

## 开发约定

### 三个评测集的区分

- **kernelbench/**：基于上游 KernelBench 格式适配的评测任务，面向跨框架对齐
- **akg_kernels_bench/**：AKG 自建的高频场景，按 shape 类型（静态/动态）和算子类别组织
- **akg_kernels_bench_lite/**：轻量评测，适合快速回归验证

### 新增评测用例

1. 评测文件使用 KernelBench 格式（`class Model(nn.Module)` + `get_inputs()` + `get_init_inputs()`）
2. 放在对应难度/类别子目录下
3. 确保可独立运行验证

## 不做什么

- **不要**把性能测试脚本放在这里——归 `tests/op/bench/`
- **不要**修改 `thirdparty/KernelBench` 子模块的源码
