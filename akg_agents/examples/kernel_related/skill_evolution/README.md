# Skill Evolution 工具集

本目录包含 Skill 自进化系统的 CLI 工具和验证脚本。

## 文件说明

| 文件 | 功能 |
|------|------|
| `run_skill_evolution.py` | 四模式 Skill 生成 CLI（不依赖 Agent 框架） |
| `verify_evolved_skill.py` | 验证 evolved skill 对算子生成效果的 A/B 对比脚本 |
| `run_ab_test.py` | 批量 A/B 测试运行器（多 group 多算子） |
| `ab_test_utils.py` | A/B 测试工具函数（运行管理、日志解析、结果收集） |
| `tracking.md` | 实验结果跟踪文档 |

## Skill Evolution 模式速查

| 模式 | 场景 | 输入 |
|------|------|------|
| `search_log` | adaptive_search 跑完后，从搜索日志提取进化链 diff 生成优化经验 | 节点 logs 目录 |
| `expert_tuning` | 用户交互式调优后，从对话历史提取"建议→代码→性能"因果链 | 对话目录 |
| `error_fix` | 从失败→成功的修复记录中提取调试经验，持续追加到同一 skill | 节点 logs 目录 |
| `merge_skills` | evolved 目录下 skill 过多时，按主题合并去重 | DSL 名称 |

## 快速上手

### 1. 生成 Skill

```bash
cd akg_agents

# 从搜索日志生成
python examples/kernel_related/skill_evolution/run_skill_evolution.py search_log /path/to/logs relu

# 从人工调优生成
python examples/kernel_related/skill_evolution/run_skill_evolution.py expert_tuning ~/.akg/conversations/cli_xxx relu

# 从错误修复生成
python examples/kernel_related/skill_evolution/run_skill_evolution.py error_fix /path/to/logs matmul

# 合并已有 skill
python examples/kernel_related/skill_evolution/run_skill_evolution.py merge_skills triton_cuda
```

### 2. 验证 Skill 效果

`verify_evolved_skill.py` 支持通过 `--skill-paths` 指定要验证的 skill（文件、目录或多个混合），默认 DSL 为 `triton_ascend`。

```bash
# 验证单个 skill 文件
python examples/kernel_related/skill_evolution/verify_evolved_skill.py \
    --task-file /path/to/op_task.py \
    --skill-paths op/resources/skills/triton-ascend/evolved/triton-ascend-error-fix/SKILL.md

# 混合多个路径
python examples/kernel_related/skill_evolution/verify_evolved_skill.py \
    --task-file /path/to/op_task.py \
    --skill-paths /path/to/error-fix/SKILL.md /path/to/exp-skill/

# 指定其他 DSL/backend，仅跑 B 组
python examples/kernel_related/skill_evolution/verify_evolved_skill.py \
    --task-file /path/to/op_task.py \
    --skill-paths /path/to/SKILL.md \
    --dsl triton_cuda --backend cuda --mode B
```

默认使用 `--task-type profile` 同时验证正确性和性能（Speedup / GenTime），如只需正确性验证可传 `--task-type precision_only`。

验证原理：使用 `kernelgen_only_workflow`，A 组仅加载 `guides/` 下原始 skill，B 组额外强制注入指定 skill（跳过选择逻辑），对比两组的 Speedup 和 GenTime。

### 3. 批量 A/B 测试

```bash
# 运行 Group 1 的批量对比
python examples/kernel_related/skill_evolution/run_ab_test.py --group 1 --mode both --device 0
```

详细文档参见 `docs/v2/CN/SkillEvolution.md`。
