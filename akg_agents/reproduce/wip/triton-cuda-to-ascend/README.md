# Triton-CUDA → Triton-Ascend 跨平台算子生成

将 SGLang / vLLM 的 triton_cuda 算子改写为 triton_ascend，通过预生成参考数据实现跨平台验证。

## 整体流程

```
┌─────────────────────────────────┐      scp / rsync     ┌──────────────────────────────────┐
│      CUDA 服务器 (A100)          │ ──────────────────── │      Ascend 服务器 (910B)          │
│                                 │     .pt 文件          │                                  │
│  1. gen_reference_cache.py      │                      │  2. run_adaptive_search_with_    │
│     ↓                           │                      │     cache.py                     │
│  遍历 sglang/vllm 算子           │                      │     (adaptive_search, 多轮进化     │
│     ↓                           │                      │      搜索 + 性能优化)              │
│  对每个算子:                     │                      │     ↓                             │
│    - import Model               │                      │  验证: .pt inputs → compare outputs│
│    - get_inputs() → forward()   │                      │  Profiling: NPU 实际跑性能         │
│    - save inputs + outputs      │                      │     ↓                             │
│     ↓                           │                      │  PASS/FAIL + speedup              │
│  ~/.akg/.tmp/reference_data/    │                      │                                  │
│    triton_cuda_cache/           │                      │                                  │
│    ├── manifest.json            │                      │                                  │
│    ├── sglang/*.pt              │                      │                                  │
│    ├── vllm_triton/*.pt         │                      │                                  │
│    └── vllm_torch/*.pt          │                      │                                  │
└─────────────────────────────────┘                      └──────────────────────────────────┘
```

## Step 1: 在 CUDA 服务器生成参考数据

### 前置条件

- CUDA GPU 可用
- `source env.sh`
- thirdparty 子模块已初始化: `git submodule update --init "akg_agents/thirdparty/*"`

### 运行

```bash
# 生成全部（sglang + vllm，约 90 个算子）
python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py

# 只生成 sglang
python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py --source sglang

# 只生成 vllm triton 算子
python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py --source vllm_triton

# 指定算子
python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py \
    --source sglang --ops triton_tanh merge_state_triton

# 指定输出目录和设备
python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py \
    --output-dir /data/ref_cache --device 1
```

### 产出

```
~/.akg/.tmp/reference_data/triton_cuda_cache/
├── manifest.json           # 包含全部算子的生成结果清单
├── sglang/
│   ├── triton_tanh.pt      # 每个 .pt 包含 inputs + outputs + init_inputs
│   ├── merge_state_triton.pt
│   └── ...                 # ~46 个算子
├── vllm_triton/
│   ├── rms_norm_kernel.pt
│   └── ...                 # ~28 个算子
└── vllm_torch/
    ├── silu_and_mul.pt
    └── ...                 # ~17 个算子
```

每个 `.pt` 文件内容（`torch.load` 后的 dict）:

```python
{
    'op_name': 'triton_tanh',
    'seed': 0,
    'save_inputs': True,
    'inputs': [tensor_cpu, ...],       # 原始输入（forward 前 clone，避免 in-place 污染）
    'input_shapes': [(16, 16384), ...],
    'input_dtypes': ['torch.float32', ...],
    'outputs': [tensor_cpu, ...],      # 参考输出
    'output_shapes': [(16, 16384), ...],
    'output_dtypes': ['torch.float32', ...],
    'init_inputs': [...],              # Model.__init__ 参数
}
```

### 拷贝到 Ascend 服务器

```bash
scp -r ~/.akg/.tmp/reference_data/triton_cuda_cache \
    ascend-server:~/.akg/.tmp/reference_data/triton_cuda_cache
```

## Step 2: 在 Ascend 上自适应搜索生成 triton_ascend 算子

### 前置条件

- Ascend NPU 可用
- `source env.sh`
- API key 已配置
- .pt 缓存已拷贝到本机

### 单算子模式

```bash
# 按 source + op 定位
python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \
    --source sglang --op triton_tanh

# 多设备 + 更多搜索轮次
python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \
    --source sglang --op merge_state_triton \
    --devices 0 1 2 3 --max-concurrent 4 --max-tasks 40

# 手动指定文件路径
python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \
    --ref-pt /path/to/merge_state_triton.pt \
    --benchmark-file benchmark/akg_kernels_bench/thirdparty/sglang/merge_state_triton.py
```

### 批量模式（断点续存）

不指定 `--op` 即进入批量模式，串行遍历 cache 目录下所有 `.pt`，逐个跑 adaptive_search。
进度记录在脚本同级 `.tmp/batch_progress.json`，重跑自动跳过已完成和已失败的 case。

```bash
# 批量跑某个 source 下所有 cache
python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \
    --source sglang

# 批量跑全部 source（sglang + vllm_triton + vllm_torch，约 91 个算子）
python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py

# 中途 Ctrl+C 或异常退出后，直接重跑同样的命令即可续跑
# 已完成（status=done）的 case 会自动跳过
```

#### 进度文件格式

```
reproduce/wip/triton-cuda-to-ascend/.tmp/
└── batch_progress.json
```

```json
{
  "cases": {
    "sglang/triton_tanh": {
      "status": "done",
      "source": "sglang",
      "op_name": "triton_tanh",
      "finished_at": "2026-04-08T15:30:00",
      "elapsed_time": 120.5,
      "result": { "total_success": 8, "best_speedup": 1.35, "..." }
    },
    "sglang/merge_state_triton": {
      "status": "error",
      "error": "RuntimeError: ...",
      "..."
    }
  },
  "summary": { "total": 46, "succeeded": 30, "failed": 2, "skipped": 32 }
}
```

#### 重跑某个 case

已完成（`done`）和已失败（`error`）的 case 都会被跳过。
想重跑某个 case，手动删掉 JSON 中对应的条目再跑即可。
想全部重跑，删掉整个 `.tmp/batch_progress.json`。

## 原理说明

### 为什么需要两步？

原始 triton_cuda 代码（如 `import triton`、CUDA kernel）在 Ascend 环境上无法 import，
因此无法直接调用 `get_inputs()` / `Model.forward()` 来获取参考输入和输出。

通过预生成参考数据 (.pt)，将平台依赖解耦：

1. **CUDA 端**: 运行原始代码，保存 inputs + outputs 到 .pt（所有 tensor 已 `.cpu()`）
2. **Ascend 端**: 加载 .pt，跳过原始 framework 代码的 import，直接用保存的 inputs 测试 LLM 生成的 triton_ascend 代码

### 关键配置项

每个算子运行时的 config 会注入以下参考数据字段：

```python
config["use_reference_data"] = True      # 启用参考数据模式
config["use_reference_inputs"] = True    # 同时使用参考输入（不调用 get_inputs()）
config["reference_data"] = ref_bytes     # .pt 文件的 bytes
```

这会触发验证模板中的条件分支：

- 不 import framework 文件（避免 CUDA 依赖）
- 从 .pt 加载 `init_inputs`（用于 `ModelNew(*init_inputs)`）
- 从 .pt 加载 `inputs`（用于 `ModelNew.forward(*inputs)`）
- 从 .pt 加载 `outputs`（作为比对基准）

### in-place 安全

`generate_reference_data(save_inputs=True)` 在 `model.forward()` **之前** clone inputs，
确保即使 forward 有 in-place 操作（如 `x.add_()`），保存的 inputs 也是原始值。

## 推荐工作流

```bash
# 1. CUDA 上生成参考数据
python gen_reference_cache.py --source sglang

# 2. scp 到 Ascend
scp -r ~/.akg/.tmp/reference_data/triton_cuda_cache ascend:~/.akg/.tmp/reference_data/

# 3. 批量跑自适应搜索（断点续存，可随时中断重跑）
python run_adaptive_search_with_cache.py --source sglang --max-tasks 30

# 4. 对重点算子加大搜索力度
python run_adaptive_search_with_cache.py --source sglang --op triton_tanh --max-tasks 60
```

## 手动使用 .pt 缓存（编程接口）

如果不想用批量脚本，可以在自己的代码中直接使用：

```python
import asyncio
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker

async def convert_single_op():
    await register_local_worker([0], backend="ascend", arch="ascend910b4")

    with open("~/.akg/.tmp/reference_data/triton_cuda_cache/sglang/triton_tanh.pt", "rb") as f:
        ref_bytes = f.read()

    with open("benchmark/akg_kernels_bench/thirdparty/sglang/triton_tanh.py") as f:
        framework_code = f.read()

    config = load_config(dsl="triton_ascend", backend="ascend")
    config["use_reference_data"] = True
    config["use_reference_inputs"] = True
    config["reference_data"] = ref_bytes
    config["max_step"] = 10

    task = LangGraphTask(
        op_name="triton_tanh",
        task_desc=framework_code,
        task_id="manual_001",
        backend="ascend", arch="ascend910b4",
        dsl="triton_ascend",
        config=config, framework="torch",
        workflow="coder_only_workflow",
    )

    op_name, success, info = await task.run()
    print(f"{'PASS' if success else 'FAIL'}: {op_name}")
    if success:
        print(info.get("coder_code", "")[:200])

asyncio.run(convert_single_op())
```

## 已覆盖算子

| 数据源 | 算子数 | 说明 |
|--------|--------|------|
| `sglang` | ~46 | SGLang triton kernel（不含 class_method） |
| `vllm_triton` | ~28 | vLLM triton kernel |
| `vllm_torch` | ~17 | vLLM torch op（激活函数、归一化、RoPE 等） |
| **合计** | **~91** | |
