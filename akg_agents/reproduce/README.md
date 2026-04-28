# reproduce — 复现脚本目录

本目录提供让外部用户复现 AKG Agents 流程效果的独立脚本。

## 目录结构

```
reproduce/
├── SPEC.md              # 本目录开发规范
├── README.md            # 本文件
├── RESULT_TEMPLATE.md   # 复现结果记录模板
└── wip/                 # 临时建设中的脚本（详见 wip/README.md）
    ├── _common.py
    ├── reproduce_mhc_coder_only.py
    ├── reproduce_mhc_kernelgen_skill.py
    ├── reproduce_kernelbench_coder_only.py
    ├── reproduce_kernelbench_kernelgen_skill.py
    ├── reproduce_akgbench_coder_only.py
    ├── reproduce_akgbench_kernelgen_skill.py
    ├── reproduce_adaptive_search.py
    └── reproduce_evolve.py
```

## 使用说明

### 运行前准备

1. 确保已安装 akg_agents：
   ```bash
   pip install -e ./ --no-build-isolation
   # 或
   source env.sh
   ```

2. 配置必要的环境变量（如 API Key）：
   ```bash
   export AKG_AGENTS_API_KEY=your_key_here
   # 或在 ~/.akg/settings.json 中配置
   ```

3. 下载第三方 benchmark：
   ```bash
   bash akg_agents/download.sh --with_kernelbench --with_evokernel
   ```

### 运行脚本

每个脚本都支持 `--help` 查看详细参数：

```bash
python reproduce/wip/<script_name>.py --help
```

### 当前可用脚本（wip/）

8 个独立脚本，分为基础 workflow 和搜索/进化策略两类：

**基础 workflow（脚本 1-6）** — 每个脚本固定使用对应 workflow：

| 脚本 | Benchmark | Workflow | 特有参数 |
|------|-----------|---------|---------|
| `reproduce_mhc_coder_only.py` | EvoKernel MHC | coder_only | `--op 5` 指定序号 |
| `reproduce_mhc_kernelgen_skill.py` | EvoKernel MHC | kernelgen | `--op 5` 指定序号 |
| `reproduce_kernelbench_coder_only.py` | KernelBench Level1 | coder_only | `--tasks`；`--include-conv` |
| `reproduce_kernelbench_kernelgen_skill.py` | KernelBench Level1 | kernelgen | `--tasks`；`--include-conv` |
| `reproduce_akgbench_coder_only.py` | AKGBench Lite | coder_only | `--tiers 1`；`--cases` |
| `reproduce_akgbench_kernelgen_skill.py` | AKGBench Lite | kernelgen | `--tiers 1`；`--cases` |

**搜索/进化策略（脚本 7-8）** — 通过 `--config` 加载 YAML 配置，`--benchmark` 选择测试集：

| 脚本 | 策略 | 配置文件 | CLI 覆盖 |
|------|------|---------|---------|
| `reproduce_adaptive_search.py` | Adaptive Search | `adaptive_search_config.yaml` | `--max-total-tasks` 等 |
| `reproduce_evolve.py` | Evolve | `evolve_config.yaml` | `--max-rounds` 等 |

通用参数：`--device`、`--pass-n`、`--profile`、`--llm-concurrency`、`--output`。

报告默认保存到 `~/.akg/reproduce_log/`。详细说明见 [wip/README.md](wip/README.md)。

## wip/ 子目录

`wip/` 目录存放**临时建设中的脚本**，这些脚本：
- 可能尚未完全验证
- 可能包含实验性功能
- 文档可能不完整

验证通过后会移至 `reproduce/` 根目录。

## 贡献指南

添加新的复现脚本时，请确保：
1. 脚本头部或配套 README 说明复现目标、前置条件、运行方式
2. 支持 `--help` 参数
3. JSON 输出包含 `benchmark` 字段，使用 `stats.op_results` 格式
4. 支持 `--pass-n` / `--profile` / `--llm-concurrency` 通用参数
5. 输出清晰，关键指标可视化
6. 先放在 `wip/` 验证，通过后移至根目录
