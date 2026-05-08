# Dynamic Tune

## 1. 用途

`akg_agents.op.dynamic_tune` 提供面向 Triton Ascend 动态 shape 场景的离线调优与在线选择能力。调优阶段显式调用 `tune_configs(...)`，生产路径只读取已落盘的 `manifest.json` 并按 selector 选择配置。

适用场景：

| 场景 | 说明 |
|---|---|
| 动态 shape kernel | 同一个 kernel 会被多组 shape 反复调用，需要按 shape 选择不同 block 参数 |
| NPU 生产路径 | 运行时不能接受在线 benchmark 带来的延迟和抖动 |
| ModelNew 后处理 | 将生成的 `ModelNew.forward(..., config=None)` 接到已部署 selector |

## 2. 核心流程

```
候选 Config
    │
    ▼
compile_gate 先编译/试跑，剔除不可用配置
    │
    ▼
BatchProfiler 对代表性 shapes × kept configs 测时
    │
    ▼
selector.fit 训练 shape -> config 的选择策略
    │
    ▼
dump_manifest 写入源码 hash 对应的 manifest 目录
    │
    ▼
生产调用时 load_deployed_selector()
```

调优阶段由 `tune_configs(...)` 触发；ST case / CLI 封装了 case 加载、调优调用、验证和 profile。生产阶段只根据 shape 调用 selector 返回显式 config。

## 3. Public API

| API | 位置 | 说明 |
|---|---|---|
| `Config` | `akg_agents.op.dynamic_tune` | 与 `triton.Config` 兼容的候选配置表达 |
| `tune_configs` | `akg_agents.op.dynamic_tune` | 显式执行离线调优，写出 `manifest.json` |
| `load_deployed_selector` | `akg_agents.op.dynamic_tune` | 按调用方源码 hash 加载生产 selector |
| `postprocess_case` / `run_case` | `akg_agents.op.dynamic_tune.cases` | case 级后处理与运行入口，供集成测试和批处理脚本复用 |

## 4. 显式调优用法

```python
from akg_agents.op.dynamic_tune import Config, tune_configs

outcome = tune_configs(
    axis_names=("M",),
    shapes=[(64,), (128,), (256,), (512,)],
    configs=[
        Config({"BLOCK_M": 32}, num_warps=4),
        Config({"BLOCK_M": 64}, num_warps=4),
    ],
    module=model_new,
    inputs_by_shape={
        (64,): inputs_for_64,
        (128,): inputs_for_128,
        (256,): inputs_for_256,
        (512,): inputs_for_512,
    },
    cache_dir="/path/to/cache",
    selector="tree",
    device="npu:0",
    warmup=1,
    repeat=10,
)
```

调用路径：

- 调优前：准备候选 `Config`、代表性 `shapes`、`ModelNew` 实例以及每个 shape 的输入。
- 调优中：`tune_configs(...)` 执行 compile gate、批量测时、训练 selector，并写入 manifest。
- 调优后：`ModelNew.forward(..., config=None)` 通过 `load_deployed_selector()` 按源码 hash 定位 manifest，按 shape 选择配置。

## 5. ModelNew 后处理约定

后处理后的 `ModelNew` 应保持：

```python
class ModelNew:
    def forward(self, X, config=None):
        if config is None:
            config = self._select_config((int(X.shape[0]),))
        block_m = config.param("BLOCK_M")
        ...
```

约束：

- `forward(..., config=None)` 必须保留，便于调优阶段显式注入 config。
- `config is None` 时才 lazy load selector；显式 config 直接使用。
- 显式 config 使用 `config.param("...")` 读取候选参数。
- `config.param("...")` 只用于 `sample.json::config_param_names` 声明的字段；其它硬件常量保持原实现来源。

这些约束由 `akg_agents.op.dynamic_tune.cases.contract._ModelNewContractValidator` 静态检查。

## 6. Manifest

调优产物位于源码 hash 对应的 manifest 目录。它包含：

| 字段 | 说明 |
|---|---|
| `schema_version` | manifest schema 版本 |
| `axis_names` | shape 维度名，与 `tune_configs(axis_names=...)` 对齐 |
| `candidates` | 全部候选 config，包含 kept / rejected 状态与拒绝原因 |
| `selector` | selector 类型、payload、运行时依赖和可选 config id |
| `tune_meta` | 测时路径、warmup、repeat、备注 |
| `extras` | 调用方附加信息 |

manifest 不记录 baseline 时延或加速比；它只描述动态选择策略。

## 7. ST Case 入口位置

Dynamic Tune 的 case 级后处理和运行入口属于运行包的一部分：

```
akg_agents/python/akg_agents/op/dynamic_tune/cases/
├── convert.py   # 调用 opencode 后处理 ModelNew，并做 contract 校验
├── runner.py    # 加载 case、调优、验证、profile、生成 summary/report
└── ...
```

ST case 和入口仍放在测试目录：

```
akg_agents/tests/op/st/dynamic_tune/
├── cases/
│   ├── original/<case>/    # 原始 base.py / impl.py / sample.json
│   └── converted/<case>/   # 后处理后的 impl.py
├── test_dynamic_tune.py
```

`test_dynamic_tune.py` 从 `akg_agents.op.dynamic_tune.cases` 导入 case 级入口。

## 8. 常用命令

只检查后处理产物 contract：

```bash
PYTHONPATH=akg_agents/python \
DYNAMIC_TUNE_IMPL_CODE=akg_agents/tests/op/st/dynamic_tune/cases/converted/relu/impl.py \
python3 -m pytest akg_agents/tests/op/st/dynamic_tune/test_dynamic_tune.py::test_convert -q
```

运行 dynamic_tune 单测：

```bash
PYTHONPATH=akg_agents/python \
python3 -m pytest akg_agents/tests/op/ut/dynamic_tune
```

运行已后处理产物（需要真实 NPU 环境）：

```bash
PYTHONPATH=akg_agents/python \
python3 -m pytest akg_agents/tests/op/st/dynamic_tune/test_dynamic_tune.py::test_run -q
```

后处理并运行（需要 opencode 和真实 NPU 环境）：

```bash
PYTHONPATH=akg_agents/python \
python3 -m pytest akg_agents/tests/op/st/dynamic_tune/test_dynamic_tune.py::test_convert_and_run -q
```

## 9. 注意事项

- `axis_names` 的顺序必须与传给 `tune_configs(shapes=...)` 的 shape tuple 顺序一致。
- `cache_dir` 是部署边界；生产环境应随代码一起发布对应 manifest。
- 如果 `compile_gate` 剔除了全部 config，调优会失败并输出每个 config 的拒绝原因。
- full validate / profile 依赖真实 NPU、torch_npu、msprof 等运行环境；普通单测不要求 NPU。
