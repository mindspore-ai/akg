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

在初始化 `KernelVerifier` 时，设置 `bench_type="sol"`，并在 `config` 中提供 `sol_problem_dir` 指向具体的 case 目录：

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
