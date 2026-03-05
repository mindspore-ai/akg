---
name: op-gen
description: >
  调用 akg_agents workflow 生成优化算子。
  提供脚本直接调用和 akg_cli 两种方式。
argument-hint: >
  必需：workflow 类型、任务文件路径、framework、backend、arch、dsl。
  可选：output-path、devices、workflow 特定参数。
---

# 算子生成

## 调用方式选择

| 方式 | 适用场景 |
|------|---------|
| **脚本**（`@scripts/run_workflow.py`） | 已明确 workflow 类型和全部参数，直接执行 |
| **akg_cli op** | 需要 akg_cli 特有功能（如 `--resume` 恢复会话、`--worker_url` 远程 Worker）|

**默认使用脚本**。仅当需要上述 akg_cli 特有功能时才切换到 akg_cli。

---

## 方式 A: 脚本调用

脚本路径：`@scripts/run_workflow.py`

### 用法

```bash
python @scripts/run_workflow.py \
  --workflow <kernelgen|adaptive_search|evolve> \
  --task-file /abs/path/{op_name}.py \
  --framework <framework> --backend <backend> \
  --arch <arch> --dsl <dsl> \
  --devices <device_ids> \
  --output-path /abs/path/output_dir
```

- `op_name` 自动从 `--task-file` 文件名推断（如 `relu.py` → `relu`），也可用 `--op-name` 显式指定。
- `--devices` 可选，不指定则自动检测。

> ⚠️ 此脚本必须在激活的环境中运行，使用 `$HOME_DIR/.akg/check_env.md` 中的命令模板包裹。

### Workflow 说明

| workflow | 特点 | 典型耗时 |
|----------|------|---------|
| `kernelgen` | KernelGen→Verifier→Conductor 迭代，适合需求明确场景（**默认**） | 1-5 分钟 |
| `adaptive_search` | UCB 策略、即时递补、收敛快 | 10-30 分钟 |
| `evolve` | 岛屿模型、多样性强，需多设备并行 | 15-60 分钟 |

**🛑 使用 `question` 工具让用户确认使用的 workflow 类型**

### 输出目录

`--output-path` 应指向 `<工作目录>/output/{workflow}_{n}/`，n 为下一可用序号（从 0 开始），避免多次运行相互覆盖。

脚本在 output-path 下生成：
- `generated_code.py`：最佳生成代码
- `summary.json`：执行摘要（workflow、success、profile 等）
- `logs/`：详细日志

### ⚠️ 长时间运行

Workflow 运行时间较长（见上方典型耗时）。bash 工具默认超时 120 秒，不足以完成 workflow。

**方案：前台执行 + 延长 `timeout` + `tee` 记录日志**

- 前台执行：输出实时可见，用户按 Esc 可随时中断（进程同步终止）
- `timeout` 参数：在 bash 调用中设置足够的超时时间（毫秒）
- `tee`：同时输出到终端和日志文件，便于事后查看

| workflow | 推荐 timeout（毫秒） |
|----------|---------------------|
| `kernelgen` | 3600000（60 分钟） |
| `adaptive_search` | 7200000（120 分钟） |
| `evolve` | 14400000（240 分钟） |

> ⛔ 禁止使用 `nohup`/`&` 后台执行——用户按 Esc 无法终止后台进程，违背用户意图。

**示例**（conda）：

```bash
# bash 工具调用时设置 timeout=7200000
conda run -n $CONDA_ENV --no-capture-output bash -c \
  "cd $AKG_AGENTS_DIR && source env.sh && \
   python <@scripts/run_workflow.py 的绝对路径> \
   --workflow adaptive_search \
   --task-file /abs/path/{op_name}.py \
   --framework torch --backend cuda \
   --arch a100 --dsl triton_cuda \
   --output-path /abs/path/output/{workflow}_{n} \
   2>&1 | tee /abs/path/output/{workflow}_{n}/run.log"
```

**示例**（venv）：

```bash
# bash 工具调用时设置 timeout=7200000
bash -c "source $VENV_PATH/bin/activate && \
  cd $AKG_AGENTS_DIR && source env.sh && \
  python <@scripts/run_workflow.py 的绝对路径> \
  --workflow adaptive_search \
  --task-file /abs/path/{op_name}.py \
  --framework torch --backend cuda \
  --arch a100 --dsl triton_cuda \
  --output-path /abs/path/output/{workflow}_{n} \
  2>&1 | tee /abs/path/output/{workflow}_{n}/run.log"
```

**完成判定**：
- 命令正常退出 + `summary.json` 存在 → 成功，读取结果
- 命令正常退出但无 `summary.json` → 失败，查看 `run.log` 获取错误信息
- 用户 Esc 中断 → 进程终止，可查看 `run.log` 了解已有进度

> ⚠️ bash 工具对输出有长度限制（约 50k token），超长输出会被截断（末尾显示 `...` 或 `[truncated]`）。**结果判定和错误分析必须读取 `run.log` 文件**，不要依赖 bash 工具返回的终端输出。

### Workflow 特定参数

**adaptive_search**:
- `--max-concurrent`：最大并发数（默认 2）
- `--initial-tasks`：初始任务数（默认 2）
- `--max-tasks`：最大总任务数（默认 10）

**evolve**:
- `--max-rounds`：进化轮数（默认 3）
- `--parallel-num`：每轮并行数（默认 4）
- `--num-islands`：岛屿数量（默认 2）

**kernelgen**:
无额外参数，使用默认迭代配置。

---

## 方式 B: akg_cli 调用

1. 执行 `akg_cli op --help` 获取最新参数说明
2. 根据 workflow 确定 `--intent`（akg_cli 通过 intent 关键词选择工作流）：
   - adaptive_search → intent 包含 "adaptive_search_workflow"
   - kernelgen → intent 包含 "kernelgen_workflow"
   - evolve → intent 包含 "evolve_workflow"
3. 构建命令（同样需延长 bash `timeout`，参照上方「长时间运行」章节）：

```bash
akg_cli op --task-file /abs/path/{op_name}.py \
  --intent '使用 adaptive_search_workflow 生成高性能算子' \
  --framework <framework> --backend <backend> \
  --arch <arch> --dsl <dsl> \
  --output-path /abs/path/output_dir \
  --yes --no-stream
```

> ⚠️ **akg_cli 注意事项**:
> - 必须同时传 `--task-file` 和 `--intent`
> - 必须使用 `--yes` 和 `--no-stream`（headless 模式）
> - 禁止使用 `echo 'y' |` 管道代替 `--yes`
> - 所有路径使用绝对路径
