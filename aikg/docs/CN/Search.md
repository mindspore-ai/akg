# AIKG 自适应搜索 (Adaptive Search)

## 1. 概述

自适应搜索模块是一个基于 **UCB (Upper Confidence Bound)** 选择策略的异步流水线搜索框架，用于替代原有的岛屿/精英进化算法。

### 1.1 核心特点

| 特性 | evolve (岛屿/精英) | adaptive_search |
|------|-------------------|-----------------|
| 执行模式 | 同步轮次（等待所有任务完成） | **异步流水线**（任务完成立即补充） |
| 父代选择 | 精英池 + 随机选择 | **UCB 选择**（性能 + 探索平衡） |
| 失败处理 | 保留信息 | **丢弃**（只保留成功任务） |

### 1.2 设计目标

1. **提高资源利用率**：异步流水线，无等待浪费
2. **智能父代选择**：UCB 策略平衡探索与利用
3. **简化逻辑**：只存成功任务，丢弃失败任务
4. **持续探索**：DB 为空时继续生成初始任务

---

## 2. 架构设计

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                        搜索控制器 (Controller)                    │
└─────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   任务池       │   │   等待队列     │   │   Success DB  │
│  (运行中任务)   │   │  (待运行任务)  │   │  (成功任务库)  │
│  max=并发数    │   │  FIFO 队列    │   │  UCB统计信息   │
└───────────────┘   └───────────────┘   └───────────────┘
        │                    ▲                    │
        │                    │                    │
        ▼                    │                    ▼
┌───────────────┐            │           ┌───────────────┐
│  任务完成      │────────────┘           │  UCB 选择器   │
│  + 性能测试    │                        │  (性能+次数)   │
└───────────────┘                        └───────────────┘
        │                                        │
        ▼                                        ▼
   成功 → 加入 DB                         ┌───────────────┐
   失败 → 丢弃                            │  任务生成器    │
                                         │  - 层次化灵感   │
                                         │  - meta_prompts │
                                         │  - handwrite    │
                                         └───────────────┘
```

### 2.2 文件结构

```
ai_kernel_generator/core/adaptive_search/
├── __init__.py           # 模块导出
├── success_db.py         # SuccessDB, SuccessRecord - 成功任务数据库
├── task_pool.py          # AsyncTaskPool, PendingTask, TaskResult - 异步任务池
├── ucb_selector.py       # UCBParentSelector - UCB 父代选择器
├── task_generator.py     # TaskGenerator - 任务生成器（复用现有组件）
├── controller.py         # AdaptiveSearchController - 搜索控制器
└── adaptive_search.py    # 主入口函数
```

---

## 3. UCB 选择策略

### 3.1 UCB 公式

$$UCB(s) = Q(s) + c \cdot \sqrt{\frac{\ln N_{total}}{N(s) + 1}}$$

其中：
- **Q(s)**: 质量得分，基于性能计算，gen_time 越小越好
- **N(s)**: 该节点被选择的次数
- **N_total**: 全局总选择次数
- **c**: 探索系数（默认 √2 ≈ 1.414）

### 3.2 质量得分计算

$$Q(s) = \frac{baseline}{baseline + gen\_time}$$

其中 baseline 是当前最佳 gen_time。

### 3.3 选择示例

```
DB 中的记录：
┌────────────────────────────────────────────────────────────────┐
│ ID     │ gen_time │ selection_count │ Q(s)  │ explore │ UCB   │
├────────────────────────────────────────────────────────────────┤
│ task_1 │ 0.5ms    │ 5               │ 0.67  │ 0.42    │ 1.09  │ ← 性能好但选太多
│ task_2 │ 0.8ms    │ 1               │ 0.56  │ 0.89    │ 1.45  │ ← UCB最高，被选中
│ task_3 │ 1.2ms    │ 0               │ 0.45  │ 1.20    │ 1.65  │ ← 未被选过，探索分高
│ task_4 │ 0.6ms    │ 3               │ 0.63  │ 0.58    │ 1.21  │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. 主循环流程

```
┌─────────────────────────────────────────────────────────────────┐
│                           初始化阶段                              │
│  1. 生成 initial_task_count 个初始任务（无灵感）                    │
│  2. 填充任务池（剩余进等待队列）                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                            主循环                                │
│                                                                  │
│  1. 等待任意任务完成                                               │
│                                                                  │
│  2. 处理完成的任务                                                 │
│     - 成功 → 加入 Success DB                                      │
│     - 失败 → 丢弃                                                 │
│                                                                  │
│  3. 补充任务池                                                    │
│     a) 等待队列有任务 → 取出提交                                    │
│     b) DB 非空 → UCB 选父代 → 层次化灵感采样 → 生成进化任务           │
│     c) DB 为空 → 生成初始任务（继续探索）                            │
│                                                                  │
│  4. 检查停止条件                                                   │
│     - 达到 max_total_tasks                                        │
│     - 达到 target_success_count                                   │
│     - 达到 target_speedup                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                           收集结果                               │
│  返回最佳实现、统计信息等                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. 配置参数

### 6.1 并发控制

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_concurrent` | int | 8 | 任务池最大并发数 |
| `initial_task_count` | int | 8 | 初始生成的任务数 |
| `tasks_per_parent` | int | 1 | 每次选择父代后生成的任务数 |

### 6.2 UCB 选择参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `exploration_coef` | float | 1.414 | UCB 探索系数 c |
| `random_factor` | float | 0.1 | 选择时的随机扰动 |
| `use_softmax` | bool | False | 是否使用 softmax 采样 |

### 6.3 停止条件

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_total_tasks` | int | 100 | 最大总任务数 |
| `target_success_count` | int | 10 | 目标成功数 |
| `target_speedup` | float | 2.0 | 目标加速比 |

**停止条件行为**：
- `max_total_tasks` 触发：**等待**剩余任务完成后结束
- `target_success_count` 触发：**取消**正在运行和等待的任务，立即结束
- `target_speedup` 触发：**取消**正在运行和等待的任务，立即结束

### 6.4 灵感采样参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `inspiration_sample_num` | int | 3 | 灵感采样数（不含父代） |
| `handwrite_sample_num` | int | 2 | 手写建议采样数 |
| `handwrite_decay_rate` | float | 2.0 | 手写建议衰减率 |

---

## 7. 使用示例

### 7.1 基本用法

```python
from ai_kernel_generator.core.worker.manager import register_worker
from ai_kernel_generator.core.adaptive_search import adaptive_search

# 1. 注册 Worker
await register_worker(backend='cuda', arch='a100', device_ids=[0, 1])

# 2. 运行自适应搜索
result = await adaptive_search(
    op_name="my_kernel",
    task_desc=task_code,
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100",
    config=config,
    
    # 并发控制
    max_concurrent=4,
    initial_task_count=4,
    
    # UCB 参数
    exploration_coef=1.414,
    random_factor=0.1,
    
    # 停止条件
    max_total_tasks=50,
    target_success_count=5,
    target_speedup=1.5
)

# 3. 获取最佳实现
for impl in result['best_implementations']:
    print(f"gen_time: {impl['gen_time']:.4f}ms, speedup: {impl['speedup']:.2f}x")
```

### 7.2 使用配置文件

```python
from ai_kernel_generator.core.adaptive_search import adaptive_search_from_config

result = await adaptive_search_from_config(
    op_name="my_kernel",
    task_desc=task_code,
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100",
    config=config,
    search_config_path="config/adaptive_search_config.yaml"
)
```