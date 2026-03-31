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
    └── reproduce_akgbench_kernelgen_skill.py
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

3. 初始化第三方 benchmark：
   ```bash
   git submodule update --init "akg_agents/thirdparty/*"
   ```

### 运行脚本

每个脚本都支持 `--help` 查看详细参数：

```bash
python reproduce/wip/<script_name>.py --help
```

### 当前可用脚本（wip/）

6 个独立脚本，对比**固定文档导入**（coder_only_workflow）和 **Skill 系统导入**（kernelgen_only_workflow）在不同 benchmark 上的效果：

| 脚本 | Benchmark | 导入方式 | 特有参数 |
|------|-----------|---------|---------|
| `reproduce_mhc_coder_only.py` | EvoKernel MHC | 固定文档 | 默认全部；`--op 5` 指定序号 |
| `reproduce_mhc_kernelgen_skill.py` | EvoKernel MHC | Skill 系统 | 默认全部；`--op 5` 指定序号 |
| `reproduce_kernelbench_coder_only.py` | KernelBench Level1 | 固定文档 | 默认全部（排除 conv 54-87） |
| `reproduce_kernelbench_kernelgen_skill.py` | KernelBench Level1 | Skill 系统 | 默认全部（排除 conv 54-87） |
| `reproduce_akgbench_coder_only.py` | AKGBench Lite | 固定文档 | 默认全部 tier；`--tiers t1` 指定 |
| `reproduce_akgbench_kernelgen_skill.py` | AKGBench Lite | Skill 系统 | 默认全部 tier；`--tiers t1` 指定 |

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
3. 输出清晰，关键指标可视化
4. 先放在 `wip/` 验证，通过后移至根目录
