# SOL-ExecBench 集成指南

SOL-ExecBench 是 NVIDIA 开源的 GPU Kernel 评估基准，包含 235 个真实的 Kernel 计算任务。相比于 KernelBench，它具有更严格的多轮测试、动态 Shape 测试（Workloads）以及更精细的容差控制。

`akg_agents` 的 `KernelVerifier` 已经原生支持了 SOL-ExecBench 的数据格式，并可以将其无缝分发到多后端（CPU、CUDA、Ascend）进行验证。

## 1. 下载数据集

在项目根目录下运行下载脚本：

```bash
bash download.sh --with_sol_execbench
```

这会自动克隆 `nvidia/sol-execbench` 仓库到 `thirdparty/sol-execbench`，并从 HuggingFace 下载完整的数据集到 `thirdparty/sol-execbench/data/benchmark/` 目录下。数据集分为 `L1`、`L2`、`Quant` 和 `FlashInfer-Bench` 四个子目录。

## 2. 在代码中使用

在初始化 `KernelVerifier` 时，设置 `bench_type="sol"`。最直接的方式是在 `config` 中提供 `sol_problem_dir`，指向具体的 case 目录：

```python
from akg_agents.op.verifier.kernel_verifier import KernelVerifier

sol_problem_dir = "thirdparty/sol-execbench/data/benchmark/L1/001_attention_softmax_dropout_value_matmul_backward"

config = {
    "log_dir": "./logs",
    "sol_problem_dir": sol_problem_dir,
    "verify_timeout": 300
}

verifier = KernelVerifier(
    op_name="attention_backward",
    framework_code="", # SOL 模式下不需要
    framework="torch",
    dsl="triton_cuda", # 或 cpp, ascendc 等
    backend="cuda",
    arch="a100",
    config=config,
    bench_type="sol"   # 关键参数
)

# 传入生成的代码（包含 ModelNew 类）
task_info = {"coder_code": "..."}
passed, log = await verifier.run(task_info)
```

### 2.1 支持的 SOL 输入格式

Verifier 会统一归一化为 `definition.json`、`workload.jsonl`、`reference.py` 三个文件。目前支持：

- `sol_problem_dir`：已展开的 SOL case 目录
- `sol_problem_json`：包含三个文件内容的 JSON 字符串，或字段包含 `reference` 和 `workloads` 的原始 SOL-ExecBench record JSON 字符串
- `sol_task_code`：OpTaskBuilder 生成的 SOL JSON 或 Markdown 三文件内容
- `task_desc`：当 `bench_type="sol"` 时，也可以直接传上述 JSON/Markdown 内容

验证脚本会优先使用官方 `sol_execbench` runtime；如果环境未安装该包，会回退到仓库内置的轻量 `sol_runtime_fallback.py`。fallback 覆盖常见 tensor 输入生成路径，足够运行 ReLU 等基础 mock/ST case；完整 SOL 数据集的复杂 `custom_inputs_entrypoint` 仍建议安装官方 runtime。

示例：

```python
config = {
    "log_dir": "./logs",
    "sol_problem_json": raw_sol_record_json,
}

verifier = KernelVerifier(
    op_name="relu",
    framework_code="",  # SOL 文件已由 sol_problem_json 提供
    framework="torch",
    dsl="triton_cuda",
    backend="cuda",
    arch="a100",
    config=config,
    bench_type="sol",
)
```

通过 KernelAgent workflow 工具调用时，同样需要传 `bench_type="sol"`，并传入 `sol_problem_dir`、`sol_problem_json` 或 `sol_task_code` 之一。workflow 会把这些字段透传到 `KernelVerifier`。

## 3. 批量运行工具

我们提供了一个批量运行工具 `reproduce/wip/run_sol_bench_batch.py`，可以用来批量验证某个目录下的所有 SOL cases。

```bash
# 示例：批量验证 L1 目录下的所有算子（需要提供一个包含生成代码的目录）
python reproduce/wip/run_sol_bench_batch.py \
    --level L1 \
    --code-dir /path/to/your/generated/codes \
    --dsl triton_cuda \
    --backend cuda \
    --arch a100
```
