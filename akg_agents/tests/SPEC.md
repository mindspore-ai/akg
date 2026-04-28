# tests/ — 测试规范

## 职责

存放 akg_agents 的所有自动化测试。核心框架测试和算子场景测试分开管理。

## 目录结构

```
tests/
├── conftest.py        # pytest 配置钩子（日志级别、工作目录恢复）
├── ut/                # 核心单元测试（不需要 LLM / GPU）
├── st/                # 核心系统测试（需要 LLM 配置）
└── op/                # 算子相关测试（详见 op/SPEC.md）
    ├── ut/            #   算子单元测试
    ├── st/            #   算子系统测试
    ├── bench/         #   算子性能/基准测试
    └── resources/     #   测试用参考算子（relu/linear/attention 等）
```

## 开发约定

### 运行测试

```bash
cd $AKG_AGENTS_DIR && source env.sh

./run_test.sh -t ut                                                  # 核心单元测试
./run_test.sh -t st                                                  # 核心系统测试
./run_test.sh -t op-ut                                               # 算子单元测试
./run_test.sh -t op-st -m "torch and triton_cuda and cuda and a100"  # 算子系统测试
./run_test.sh -t op-bench -m "torch and triton_cuda and cuda and a100"

pytest tests/ut/test_llm_client.py -v  # 单个测试
```

### 测试类型

| 类型 | 目录 | 依赖 | 用途 |
|------|------|------|------|
| ut | `ut/`、`op/ut/` | 无外部依赖 | 纯逻辑验证 |
| st | `st/`、`op/st/` | LLM 配置 | 端到端流程验证 |
| bench | `op/bench/` | LLM + 硬件 | 性能基准 |

### Marker 约定（op-st / op-bench）

op-st 和 op-bench 必须同时指定四类 marker，否则 `run_test.sh` 拒绝执行：
- framework: `torch` / `mindspore`
- dsl: `triton_cuda` / `triton_ascend` / `cpp` 等
- backend: `cuda` / `ascend` / `cpu`
- arch: `a100` / `ascend910b4` 等

### 日志级别

通过 `AKG_AGENTS_LOG_LEVEL` 环境变量控制（`0`=DEBUG, `1`=INFO, `2`=WARNING, `3`=ERROR）。

## 不做什么

- **不要**在源码目录（如 `core_v2/tests/`）写新测试——统一放到本目录对应位置
- **不要**写依赖特定用户环境的测试（如硬编码路径）
- **不要**把 st 级别的测试放到 ut 目录（ut 不应依赖 LLM）

## 参考

- [tests/op/SPEC.md](op/SPEC.md) — 算子测试的详细规范
