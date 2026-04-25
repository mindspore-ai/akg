# tests/op/ — 算子测试规范

## 职责

算子/内核生成场景的专属测试。

## 目录结构

```
tests/op/
├── ut/            # 算子单元测试（适配器、配置、builder 等）
├── st/            # 算子系统测试（agent、worker、端到端）
├── bench/         # 算子性能/基准测试
└── resources/     # 测试用参考算子（relu、linear、attention 等 fixture）
```

## 开发约定

### ut / st / bench 的区分

| 类型 | 目录 | 依赖 | 说明 |
|------|------|------|------|
| ut | `ut/` | 无 LLM / GPU | 适配器逻辑、配置校验、工具函数等纯逻辑测试 |
| st | `st/` | LLM + 可选硬件 | Agent 端到端、workflow 集成测试 |
| bench | `bench/` | LLM + 硬件 | 生成质量和性能基准（Triton CUDA、Triton Ascend 等） |

### Marker 约定

st 和 bench 测试文件必须用 pytest marker 标注环境需求：

```python
import pytest

@pytest.mark.torch
@pytest.mark.triton_cuda
@pytest.mark.cuda
@pytest.mark.a100
def test_triton_cuda_relu():
    ...
```

运行时通过 `-m` 指定：`./run_test.sh -t op-st -m "torch and triton_cuda and cuda and a100"`

### resources/ 使用方式

`resources/` 下存放测试用的参考算子实现（KernelBench 格式），作为 fixture 供测试加载，不要直接 `import`。

## 不做什么

- **不要**把不需要 LLM 的测试放在 `st/`——归 `ut/`
- **不要**把评测集放在这里——归 `benchmark/`
- **不要**在 `resources/` 中放测试脚本——只放参考数据
