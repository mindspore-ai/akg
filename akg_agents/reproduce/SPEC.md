# reproduce/ — 复现脚本

## 职责

提供让外部用户复现 AKG Agents 流程效果的独立脚本，用于演示、验证和对比。

## 目录结构

| 子目录/文件 | 说明 |
|------------|------|
| `wip/` | 临时建设中的脚本（不保证完全验证，可能包含实验性功能） |
| 根目录脚本 | 已验证、可直接运行的复现脚本 |

## 开发约定

### 脚本规范

1. **独立可运行**：每个脚本应自包含，明确说明依赖和运行环境
2. **清晰的文档**：脚本头部或配套 README 说明：
   - 复现的目标场景/效果
   - 前置条件（硬件、软件、API Key 等）
   - 运行方式和预期输出
   - 数据/配置文件路径
3. **命令行友好**：支持 `--help` 或 `argparse`/`typer` 参数说明
4. **输出规范**：
   - 日志输出清晰，便于调试
   - 关键指标可视化或汇总输出

### wip/ 子目录

- 存放**未完全验证**或**实验性**的复现脚本
- 可能包含临时调试代码、不完整的文档
- 验证通过后移至 `reproduce/` 根目录

### 脚本命名

- 使用描述性名称，如 `reproduce_kernel_gen_triton_ascend.py`、`compare_triton_backends.sh`
- 避免通用名称如 `test.py`、`run.py`

### 依赖管理

- 如需特殊依赖，在脚本头部注释或 `requirements.txt` 中说明
- 优先使用 `akg_agents` 已有依赖，避免引入额外包

## JSON 输出规范

所有复现脚本输出的 JSON 报告**必须**遵守以下约定：

### 必选顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `benchmark` | string | Benchmark 标识，如 `KernelBench_Level1_no_Conv`、`AKGBench_Lite`、`EvoKernel_MHC` |
| `script` | string | 脚本标识名 |
| `workflow` | string | 使用的 workflow 完整名（如 `kernelgen_only_workflow`） |
| `pass_n` | int | Pass@N，每个算子独立运行的次数 |
| `ops_count` | int | 算子总数 |
| `elapsed_s` | float | 总耗时（秒） |
| `device_ids` | list[int] | 使用的设备 ID 列表 |
| `max_concurrency` | int | 设备任务并行度 |
| `llm_concurrency` | int | LLM 请求并发数 |
| `env_spec` | object | 环境规范（由 `collect_env_spec` 自动采集） |
| `task_log_dir` | string | 任务详细日志目录 |
| `stats` | object | 统计结果（见下方） |

### 可选顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `adaptive_search_config` | object | adaptive_search 搜索参数（仅 reproduce_adaptive_search.py 输出） |
| `evolve_config` | object | evolve 进化参数（仅 reproduce_evolve.py 输出） |

### stats 结构

```json
{
  "total_ops": 66,
  "passed_ops": 50,
  "failed_ops": 16,
  "pass_rate": 0.758,
  "op_results": {
    "<op_name>": {
      "passed": 1,
      "total": 1,
      "profile": { "gen_time": 12.3, "base_time": 15.1, "speedup": 1.23 }
    }
  }
}
```

- 使用 `op_results`（**不是** `op_stats`）
- `profile` 可选，仅在 profiling 成功时出现

### 通用参数要求

基础脚本（1-6）通过 `_common.add_common_args` 支持以下通用参数：

- `--device`、`--concurrency`、`--llm-concurrency`、`--arch`
- `--pass-n`、`--output`、`--profile`

adaptive_search / evolve 独立脚本通过 `--config` 加载 YAML 配置文件，CLI 参数可覆盖。

## 不做什么

- **不要**把单元测试放在这里——归 `tests/`
- **不要**把通用工具脚本放在这里——归 `tools/`
- **不要**把使用示例放在这里——归 `examples/`
- **不要**把评测集放在这里——归 `benchmark/`

## 与其他目录的区别

| 目录 | 用途 | 复现脚本的区别 |
|------|------|--------------|
| `examples/` | 教学性质的功能示例 | `reproduce/` 面向**端到端效果复现**，通常涉及完整流程 |
| `tests/` | 自动化测试（CI/CD） | `reproduce/` 面向**人工验证和演示**，输出更直观 |
| `tools/` | 开发运维辅助工具 | `reproduce/` 面向**外部用户验证效果**，文档更完善 |
| `benchmark/` | 评测数据集 | `reproduce/` 是**运行脚本**，benchmark 是**数据** |
