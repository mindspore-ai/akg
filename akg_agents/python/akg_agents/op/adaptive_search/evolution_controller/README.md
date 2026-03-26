# Evolution Controller 使用手册

## 简介

Evolution Controller（进化控制器）是 AKG 自适应搜索的**外挂增强模块**，通过三个求解器优化搜索过程：

| 求解器 | 功能 | 替代 |
|--------|------|------|
| Solver 1 | 谱系感知父代选择（软岛屿） | 原始 UCB 选择 |
| Solver 2 | Boltzmann + 迁移灵感选择 | 固定分层 3:4:3 |
| Solver 3 | 多维收敛检测 | 硬编码 max_total_tasks |

**兼容性**：默认关闭，不影响任何现有行为。

## 快速开始

### 方式 1：通过 `adaptive_search()` 函数参数启用

```python
from akg_agents.op.adaptive_search import adaptive_search

result = await adaptive_search(
    op_name="elu",
    task_desc=task_desc,
    dsl="triton_ascend",
    framework="torch",
    backend="ascend",
    arch="ascend910b4",
    config=config,
    max_total_tasks=30,
    use_evolution_controller=True,  # 启用进化控制器
)
```

### 方式 2：通过配置文件启用

在 `op/config/adaptive_search_config.yaml` 中设置：

```yaml
use_evolution_controller: true
```

进化控制器的详细参数配置在独立文件中：
`op/adaptive_search/evolution_controller/evolution_controller_config.yaml`

### 方式 3：在示例脚本中启用

修改示例脚本的 `adaptive_search()` 调用，添加 `use_evolution_controller=True` 参数。

## 工作原理

### Solver 1：谱系感知父代选择

**两阶段选择**：

1. **层间选择**：选择哪条谱系获得进化机会
   - 好的谱系获得更多机会（基于 speedup 潜力）
   - 但不能无限膨胀（探索赤字机制负反馈）
   - 温度参数控制探索-利用平衡

2. **层内选择**：在选定谱系内选择具体记录
   - 使用局部 UCB（全局排名质量 + 谱系内探索奖励）

**退火调度**：温度随搜索进度指数衰减——早期强探索，后期强利用。

### Solver 2：Boltzmann + 迁移灵感选择

- **本谱系为主**：默认从父代所在谱系内 Boltzmann 采样灵感
- **概率迁移**：每个灵感位以 20% 概率从其他谱系精英中选取
- **温度联动**：与 Solver 1 共享退火温度，减少独立参数

### Solver 3：多维收敛检测

监测四个信号：
- S1：性能是否停滞（最优 gen_time 不再改善）
- S2：多样性是否在下降（谱系分布是否在坍缩）
- S3：谱系是否被充分探索
- S4：边际收益是否已消失（新任务是否还能产生突破）

三状态机：`EXPLORING` → `WATCHING` → `STOPPED`

当进入 WATCHING 状态时，Solver 3 会向 Solver 1/2 发送策略信号，增大探索力度（提升温度、提高迁移率），尝试打破局部最优。

## 配置参数

### 父代选择参数（Solver 1）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `tau_start` | 2.0 | 温度起始值（高=强探索） |
| `tau_end` | 0.3 | 温度终止值（低=强利用） |
| `c_local_start` | 2.0 | 谱系内探索系数起始值 |
| `c_local_end` | 0.5 | 谱系内探索系数终止值 |
| `p_exploit` | 0.7 | Phase 1 中 exploit 模式概率 |
| `quality_bonus_strength` | 0.3 | 质量奖励强度 |

### 灵感选择参数（Solver 2）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `inspiration_sample_num` | 3 | 灵感采样数 |
| `p_migrate` | 0.2 | 跨谱系迁移概率（文献推荐 5%-20%） |
| `foreign_elite_num` | 3 | 每条外部谱系的精英数 |
| `T_start` | 2.0 | Boltzmann 温度起始值 |
| `T_end` | 0.3 | Boltzmann 温度终止值 |

### 收敛检测参数（Solver 3）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `convergence_window` | 10 | 信号计算窗口大小 |
| `perf_improvement_threshold` | 0.01 | 性能改善阈值（1%） |
| `diversity_change_threshold` | 0.05 | 多样性变化阈值 |
| `activity_threshold` | 0.8 | 谱系充分探索阈值 |
| `patience` | 2 | 停滞容忍窗口数 |
| `max_total_tasks` | 100 | 安全阀（硬性上限） |
| `max_time_seconds` | None | 时间预算（秒，可选） |

## 高级用法

### 自定义配置

```python
from akg_agents.op.adaptive_search.evolution_controller import (
    EvolutionController,
    EvolutionControllerConfig,
    ParentSelectionConfig,
    InspirationSelectionConfig,
    ConvergenceConfig,
)

config = EvolutionControllerConfig(
    parent_selection=ParentSelectionConfig(
        tau_start=3.0,      # 更强的初始探索
        tau_end=0.1,        # 更强的后期利用
        p_exploit=0.8,      # 更偏向 exploit
    ),
    inspiration=InspirationSelectionConfig(
        p_migrate=0.3,      # 更高的迁移率
    ),
    convergence=ConvergenceConfig(
        patience=5,          # 更有耐心
        max_time_seconds=1800,  # 30 分钟时间预算
    ),
)

evo = EvolutionController(config)
```

### 获取诊断信息

```python
diagnostics = evo.get_diagnostics()
print(f"多样性指数: {diagnostics['diversity_index']}")
print(f"收敛状态: {diagnostics['convergence']['state']}")
print(f"谱系数量: {diagnostics['num_lineages']}")
for root_id, stats in diagnostics['lineage_stats'].items():
    print(f"  谱系 {root_id}: size={stats['size']}, "
          f"best_speedup={stats['best_speedup']:.2f}x, "
          f"share={stats['share']:.1%}")
```

## 参数调优指南

### 高灵敏度参数（需要仔细调优）

- **tau_start / tau_end**：直接控制探索-利用平衡
  - 如果搜索容易卡在局部最优：增大 tau_start
  - 如果搜索后期性能不够好：降低 tau_end
- **patience**：过小导致过早停止，过大导致浪费

### 中灵敏度参数

- **p_exploit**：0.6-0.8 范围内影响不大
- **p_migrate**：文献推荐 0.05-0.2

### 低灵敏度参数（保持默认即可）

- **quality_bonus_strength**：0.2-0.5 均可

## 与原始 AKG 的兼容性

- `use_evolution_controller=False`（默认）：行为与修改前**完全一致**
- `use_evolution_controller=True`：启用进化控制器
- 所有原始参数（`exploration_coef`, `use_tiered_sampling` 等）在关闭进化控制器时仍然生效
- 开启进化控制器时，父代选择和灵感选择由进化控制器接管，原始 UCB 和分层采样不再使用

## 文件结构

```
evolution_controller/
├── __init__.py                  # 公共 API
├── config.py                    # 配置 dataclass
├── lineage_tree.py              # 谱系树（从 SuccessDB 重建）
├── parent_selector.py           # Solver 1: 谱系感知父代选择
├── inspiration_selector.py      # Solver 2: Boltzmann + 迁移灵感选择
├── convergence_detector.py      # Solver 3: 多维收敛检测
├── controller.py                # EvolutionController 门面类
└── README.md                    # 本文件
```
