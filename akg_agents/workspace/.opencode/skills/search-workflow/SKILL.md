---
name: search-workflow
description: >
  通过 adaptive_search 或 evolve 搜索式 workflow 生成优化算子。
  后台 silent mode 执行，轮询监控进度。
argument-hint: >
  必需：workflow 类型（adaptive_search / evolve）、任务文件路径、framework、backend、arch、dsl、输出路径、命令模板。
  可选：devices、workflow 特定参数。
---

# 搜索式算子生成（adaptive_search / evolve）

## What I do

通过 `adaptive_search` 或 `evolve` 搜索式 workflow 生成高性能算子代码。以 **silent mode** 后台执行，轮询监控进度。

## When to use me

主 Agent 的 Phase 3 中用户选择了 `adaptive_search` 或 `evolve` 时使用。

## Workflow

### 1. 启动

使用命令模板后台启动 `@scripts/run_workflow.py`，输出重定向到 `run.log`：

```bash
<命令模板: nohup python @scripts/run_workflow.py 的绝对路径 \
  --workflow <adaptive_search|evolve> \
  --task-file /abs/path/{op_name}.py \
  --framework <framework> --backend <backend> \
  --arch <arch> --dsl <dsl> \
  --output-path <output-path> \
  > <output-path>/run.log 2>&1 & echo $!>
```

记录返回的 **PID**。

### 2. 轮询监控

每隔 ~1 分钟执行：

```bash
kill -0 <PID> 2>/dev/null && echo "RUNNING" || echo "FINISHED"
tail -n 10 <output-path>/run.log
```

- `RUNNING` → 继续轮询（间隔 1 分钟）
- `FINISHED` → 进入结果收集

**禁止擅自终止工作流**

### 3. 结果收集

- `summary.json` 存在 → 读取 `success` 字段
- `generated_code.py` 存在 → 成功
- 均不存在 → 查看 `run.log` 末尾获取错误信息

### 4. 中断控制（用户要求时）

```bash
kill <PID>
```

中断后按「部分结果恢复」处理。

---

## 部分结果恢复

adaptive_search / evolve 每完成一次成功验证就会写入 `<output-path>/logs/`。进程被杀时已写入的文件仍在。

**核心原则：只要有 passed case 就不算失败。**

```
<output-path>/logs/
├── passed_cases/{op_name}/              # 通过验证的迭代
│   └── Iteration{id}_Step{NN}_verify/
│       └── {op_name}_{dsl}_impl.py      # 实现代码
├── {op_name}/profiling/
│   └── speed_up_record.txt              # 加速比记录
└── verification_results.jsonl
```

**恢复步骤**：
1. 检查 `logs/passed_cases/{op_name}/` 是否有内容
2. 读取 `speed_up_record.txt`，找性能最佳记录对应的 `unique_dir`
3. 从 `passed_cases/{op_name}/{unique_dir}/` 读取实现代码
4. 写入 `<output-path>/generated_code.py`

---

## Workflow 参数

| workflow | 特点 | 典型耗时 |
|----------|------|---------|
| `adaptive_search` | UCB 策略、即时递补、收敛快 | 10-30 分钟 |
| `evolve` | 岛屿模型、多样性强 | 15-60 分钟 |

**adaptive_search 特定参数**:
- `--max-concurrent`（默认 2）、`--initial-tasks`（默认 2）、`--max-tasks`（默认 10）

**evolve 特定参数**:
- `--max-rounds`（默认 3）、`--parallel-num`（默认 4）、`--num-islands`（默认 2）

---

## 方式 B: akg_cli 调用（替代方案）

仅当需要 `--resume` 恢复会话或 `--worker_url` 远程 Worker 时使用：

```bash
akg_cli op --task-file /abs/path/{op_name}.py \
  --intent '使用 adaptive_search_workflow 生成高性能算子' \
  --framework <framework> --backend <backend> \
  --arch <arch> --dsl <dsl> \
  --output-path <output-path> \
  --yes --no-stream
```
