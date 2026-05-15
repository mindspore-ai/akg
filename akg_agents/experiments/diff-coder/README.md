# diff-coder：对照组实验矩阵（`akg_agents/experiments/diff-coder`）

对照实验报告（实验说明摘要、观察、结论与建议；**复现与代码见 GitCode 分支**，数值基于 `summary.tsv` 聚合）：**`EXPERIMENT_REPORT.md`**。

在 **`akg_now`** 中，本目录用于：

1. **多棵代码树**（可选）：例如不同分支/目录下的 `akg_agents` 对照。  
2. **diff 修复开/关**（推荐）：同一棵树，通过 profile 的 **`repair_variants`** 注入 `AIKG_TARGETED_REPAIR=1/0`，与 **首轮构造错误**（`AIKG_FORCE_FIRST_FAIL` + `AIKG_FORCE_FAIL_MODE`）组合跑矩阵。

首轮注入与修复链路与 **`python/akg_agents/op/langgraph_op/nodes.py`**（Coder / Verifier 节点）及 **`core/agent/coder.py`**（diff 修复）一致；**benchmark 汇总里的 `avg_llm_*`** 依赖 **`utils/llm_timing_context.py`** + **`AgentBase.run_llm`** 记账 + **`LangGraphTask.run` finally** 写入 **`AIKG_BENCHMARK_LOCAL_TIMING_JSONL`**（与 `aikg_run2_no_targeted` 行为对齐）。

Conductor 说明见 **`docs/v1/Conductor.md`**（及中文版 **`docs/v1/CN/Conductor.md`**）。

## 依赖

- 宿主机 **Docker**，镜像见 profile（默认 `megalens-training:ngc25.04-proxy-tf-aikgdeps-20260426`）。
- **`akg_agents/scripts/benchmark_local_kernelbench_repeat.sh`**：会拉取 **KernelBench**（`download.sh --with_kernelbench` 或脚本内回退克隆）。
- **API**：见 **`DEEPSEEK_API_SETUP.md`**；在 `akg_now/akg` 根目录放置 **`.deepseek_api_env`**（勿提交），profile 里 **`env_file`** 指向其绝对路径。

## 快速开始（diff 修复 A/B）

```bash
cd /home/zhangyize/akg_now/akg/akg_agents/experiments/diff-coder

python3 run_matrix.py run --profile profiles/kernelbench_triton_cuda_repair_ab.json --dry-run

python3 run_matrix.py run --profile profiles/kernelbench_triton_cuda_repair_ab.json --continue-on-error

python3 run_matrix.py summarize --glob '/home/zhangyize/akg_now/akg/out/diffcoder/kernelbench_triton_cuda_repair_ab_*'
```

带 **`repair_variants`** 时，输出目录 slug 为：

`{out_dir_prefix}/{name}_{tree_label}_{repair_label}_{bench_id}_{fail_mode}/`

未配置 **`repair_variants`** 时，与旧版一致：`{name}_{tree_label}_{bench_id}_{fail_mode}/`。

## 其它 profile

- **`profiles/kernelbench_triton_cuda.json`**：从上游同步的「两树 / 多树」示例（若 `workspace_root` 仍指向旧路径，请按本机修改）。
- **`profiles/kernelbench_triton_cuda_repair_ab.json`**：**本仓库推荐**：单树 `akg_agents` × `targeted_on` / `targeted_off`。

## Profile 常用字段

| 字段 | 含义 |
|------|------|
| `workspace_root` | 含 `akg_agents/` 的仓库根（本环境为 **`/home/zhangyize/akg_now/akg`**） |
| `trees` | 对照树：`relative_path`、`label`、可选 `extra_env` |
| `repair_variants` | 可选：如 `targeted_on` / `targeted_off`，每项含 `label` 与 `extra_env`（如 `AIKG_TARGETED_REPAIR`） |
| `repair_variant_subset` | 可选，只跑部分 repair key |
| `benchmarks` | `id`、`script`、`test_target`、可选 `test_target_by_tree` |
| `fail_modes` | `syntax` / `import` / `name` / `runtime` / `type` → `AIKG_FORCE_FAIL_MODE` |
| `force_first_fail` | `true` → `AIKG_FORCE_FIRST_FAIL=1`（首轮构造错误） |
| `extra_env` | 如 `AIKG_TARGETED_REPAIR_ATTEMPTS`、`AIKG_BENCH_ROUND_TIMEOUT_S` |

Schema：**`profiles/profile.schema.json`**。

## 许可

Apache-2.0（见 `run_matrix.py` 文件头）。
