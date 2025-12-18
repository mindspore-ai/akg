# AIKG Adaptive Search

## 1. Overview

The Adaptive Search module is an asynchronous pipeline search framework based on **UCB (Upper Confidence Bound)** selection strategy, designed to replace the original island/elite evolutionary algorithm.

### 1.1 Key Features

| Feature | evolve (Island/Elite) | adaptive_search |
|---------|----------------------|-----------------|
| Execution Mode | Synchronous rounds (wait for all tasks) | **Asynchronous pipeline** (refill on completion) |
| Parent Selection | Elite pool + random | **UCB selection** (performance + exploration balance) |
| Failure Handling | Keep information | **Discard** (only keep successful tasks) |

### 1.2 Design Goals

1. **Improve Resource Utilization**: Asynchronous pipeline, no waiting waste
2. **Intelligent Parent Selection**: UCB strategy balances exploration and exploitation
3. **Simplified Logic**: Only store successful tasks, discard failures
4. **Continuous Exploration**: Generate initial tasks when DB is empty

---

## 2. Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                      Controller (搜索控制器)                      │
└─────────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐   ┌───────────────┐   ┌───────────────┐
│   Task Pool   │   │ Waiting Queue │   │   Success DB  │
│ (Running)     │   │ (Pending)     │   │ (Successful)  │
│  max=concur   │   │  FIFO Queue   │   │  UCB Stats    │
└───────────────┘   └───────────────┘   └───────────────┘
        │                    ▲                    │
        │                    │                    │
        ▼                    │                    ▼
┌───────────────┐            │           ┌───────────────┐
│ Task Complete │────────────┘           │ UCB Selector  │
│ + Profiling   │                        │ (perf+count)  │
└───────────────┘                        └───────────────┘
        │                                        │
        ▼                                        ▼
   Success → Add to DB                  ┌───────────────┐
   Failure → Discard                    │ Task Generator │
                                        │ - Tiered insp  │
                                        │ - meta_prompts │
                                        │ - handwrite    │
                                        └───────────────┘
```

### 2.2 File Structure

```
ai_kernel_generator/core/adaptive_search/
├── __init__.py           # Module exports
├── success_db.py         # SuccessDB, SuccessRecord - Success task database
├── task_pool.py          # AsyncTaskPool, PendingTask, TaskResult - Async task pool
├── ucb_selector.py       # UCBParentSelector - UCB parent selector
├── task_generator.py     # TaskGenerator - Task generator (reuses existing components)
├── controller.py         # AdaptiveSearchController - Search controller
└── adaptive_search.py    # Main entry function
```

---

## 3. UCB Selection Strategy

### 3.1 UCB Formula

$$UCB(s) = Q(s) + c \cdot \sqrt{\frac{\ln N_{total}}{N(s) + 1}}$$

Where:
- **Q(s)**: Quality score based on performance, lower gen_time is better
- **N(s)**: Number of times this node has been selected
- **N_total**: Global total selection count
- **c**: Exploration coefficient (default √2 ≈ 1.414)

### 3.2 Quality Score Calculation

$$Q(s) = \frac{baseline}{baseline + gen\_time}$$

Where baseline is the current best gen_time.

### 3.3 Selection Example

```
Records in DB:
┌────────────────────────────────────────────────────────────────┐
│ ID     │ gen_time │ selection_count │ Q(s)  │ explore │ UCB   │
├────────────────────────────────────────────────────────────────┤
│ task_1 │ 0.5ms    │ 5               │ 0.67  │ 0.42    │ 1.09  │ ← Good perf but selected too much
│ task_2 │ 0.8ms    │ 1               │ 0.56  │ 0.89    │ 1.45  │ ← Highest UCB, selected
│ task_3 │ 1.2ms    │ 0               │ 0.45  │ 1.20    │ 1.65  │ ← Never selected, high explore
│ task_4 │ 0.6ms    │ 3               │ 0.63  │ 0.58    │ 1.21  │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. Main Loop Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       Initialization Phase                       │
│  1. Generate initial_task_count initial tasks (no inspiration)   │
│  2. Fill task pool (remaining go to waiting queue)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                          Main Loop                               │
│                                                                  │
│  1. Wait for any task to complete                                │
│                                                                  │
│  2. Process completed tasks                                      │
│     - Success → Add to Success DB                                │
│     - Failure → Discard                                          │
│                                                                  │
│  3. Refill task pool                                             │
│     a) If waiting queue has tasks → Take and submit              │
│     b) If DB not empty → UCB select parent → Tiered sampling →   │
│                          Generate evolved tasks                   │
│     c) If DB empty → Generate initial tasks (continue exploring) │
│                                                                  │
│  4. Check stop conditions                                        │
│     - Reached max_total_tasks                                    │
│     - Reached target_success_count                               │
│     - Reached target_speedup                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Collect Results                           │
│  Return best implementations, statistics, etc.                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Inspiration Sampling Strategy

For evolved tasks, select **parent + tiered sampling inspirations**:

```
┌─────────────────────────────────────────────────────────────────┐
│                      UCB Select Parent                           │
│                             │                                    │
│                             ▼                                    │
│                     ┌───────────────┐                           │
│                     │    Parent     │  (is_parent=True)          │
│                     │  UCB Selected │                            │
│                     └───────────────┘                           │
│                             +                                    │
├─────────────────────────────────────────────────────────────────┤
│ Other successful impls in DB (sorted by gen_time, exclude parent)│
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   GOOD       │  │   MEDIUM     │  │   POOR       │          │
│  │   Top 30%    │  │   Mid 40%    │  │   Bottom 30% │          │
│  │  Pick 1 best │  │  Pick 1 best │  │  Pick 1 best │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│         ↓                ↓                ↓                     │
│  Final inspirations = [parent] + [good, medium, poor]           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Configuration Parameters

### 6.1 Concurrency Control

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_concurrent` | int | 8 | Maximum concurrent tasks in pool |
| `initial_task_count` | int | 8 | Number of initial tasks to generate |
| `tasks_per_parent` | int | 1 | Tasks to generate per parent selection |

### 6.2 UCB Selection Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `exploration_coef` | float | 1.414 | UCB exploration coefficient c |
| `random_factor` | float | 0.1 | Random perturbation during selection |
| `use_softmax` | bool | False | Use softmax sampling instead of argmax |

### 6.3 Stop Conditions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_total_tasks` | int | 100 | Maximum total tasks |
| `target_success_count` | int | 10 | Target success count |
| `target_speedup` | float | 2.0 | Target speedup ratio |

**Stop Condition Behavior**:
- `max_total_tasks` triggered: **Wait** for remaining tasks to complete
- `target_success_count` triggered: **Cancel** running and waiting tasks, return immediately
- `target_speedup` triggered: **Cancel** running and waiting tasks, return immediately

### 6.4 Inspiration Sampling Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `inspiration_sample_num` | int | 3 | Inspiration sample count (excluding parent) |
| `handwrite_sample_num` | int | 2 | Handwrite suggestion sample count |
| `handwrite_decay_rate` | float | 2.0 | Handwrite suggestion decay rate |

---

## 7. Usage Examples

### 7.1 Basic Usage

```python
from ai_kernel_generator.core.worker.manager import register_worker
from ai_kernel_generator.core.adaptive_search import adaptive_search

# 1. Register Worker
await register_worker(backend='cuda', arch='a100', device_ids=[0, 1])

# 2. Run adaptive search
result = await adaptive_search(
    op_name="my_kernel",
    task_desc=task_code,
    dsl="triton_cuda",
    framework="torch",
    backend="cuda",
    arch="a100",
    config=config,
    
    # Concurrency control
    max_concurrent=4,
    initial_task_count=4,
    
    # UCB parameters
    exploration_coef=1.414,
    random_factor=0.1,
    
    # Stop conditions
    max_total_tasks=50,
    target_success_count=5,
    target_speedup=1.5
)

# 3. Get best implementations
for impl in result['best_implementations']:
    print(f"gen_time: {impl['gen_time']:.4f}ms, speedup: {impl['speedup']:.2f}x")
```

### 7.2 Using Config File

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