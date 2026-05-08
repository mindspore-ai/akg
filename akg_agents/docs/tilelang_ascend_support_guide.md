# TileLang-Ascend 接入 AKG Agents Step-by-Step 指南

> 目标：让 AKG Agents 的 KernelVerifier 能够验证 TileLang-Ascend 生成的 kernel。
> 参照物：`triton_ascend` 的已有支持，做到 1:1 对应。

---

## 背景：为什么 tilelang_npuir 不可用

之前有一个 `tilelang_npuir` DSL 定义，但存在以下问题导致实际不可用：

| 问题 | 严重程度 | 说明 |
|------|:---:|------|
| `create_impl_module()` 未实现 | 🔴 P0 | 基类返回空字符串，验证脚本不会生成 `impl_model = ModelNew(...)` 代码 |
| `get_impl_import()` 格式过时 | 🔴 P0 | 使用函数式 `import {func_name}` 而非类格式 `import ModelNew` |
| `impl_func_name` 默认值不对 | 🔴 P0 | 走 else 分支生成 `{op}_{dsl}_{framework}`，而不是 `ModelNew` |
| Skills 目录完全缺失 | 🔴 P0 | `op/resources/skills/tilelang-npuir/` 不存在 |
| Prompt 缺少代码骨架 | 🟠 P1 | user_prompt.j2 / codegen.j2 中没有 tilelang 的示例代码 |

因此我们的策略是：**新建 `tilelang_ascend` DSL**（而非修 `tilelang_npuir`），命名与 `triton_ascend` 一致，用 `_ascend` 后缀。

---

## 前置条件：确认 TileLang-Ascend 环境

```bash
cd /home/yyz && source env_yyz.sh
cd /home/yyz/tilelang-ascend && source set_env.sh

# 验证环境
python -c "
import tilelang
import tilelang.language as T
print('tilelang OK')
import torch
import torch_npu
print('torch_npu OK')
"

# 跑一个简单示例确认端到端通
cd examples/gemm
python example_gemm.py
```

---

## Step 1：在 akg_agents 中确认基础环境

```bash
cd /home/yyz/akg/akg_agents
source env.sh
pip install -e ./ --no-build-isolation 2>&1 | tail -5

# 确认 akg_agents 可导入
python -c "import akg_agents; print('akg_agents OK')"
```

---

## Step 2：创建 DSL 适配器 `tilelang_ascend.py`

> 文件: `python/akg_agents/op/verifier/adapters/dsl/tilelang_ascend.py`

这是最核心的改动。参照 [triton_ascend.py](file:///home/yyz/akg/akg_agents/python/akg_agents/op/verifier/adapters/dsl/triton_ascend.py) 实现。

需实现的三个关键方法：

### 2.1 `get_import_statements()`
```python
def get_import_statements(self, framework: str) -> str:
    code = """import tilelang
import tilelang.language as T
import torch
try:
    from akg_agents.op.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""
    return code
```

### 2.2 `get_impl_import()` — 类格式（关键！）
```python
def get_impl_import(self, op_name: str, impl_func_name: str) -> str:
    return f"from {op_name}_tilelang_ascend_impl import ModelNew\n"
```

### 2.3 `create_impl_module()` — 生成实例化代码（关键！）
```python
def create_impl_module(self, framework, framework_adapter,
                       init_params_var="init_params",
                       device_var="device"):
    code = f"impl_model = ModelNew(*{init_params_var})\n"
    if framework == "torch":
        code += f"impl_model = impl_model.to({device_var})\n"
    return code
```

### 2.4 其他方法（参照 triton_ascend 抄过来）
```python
def call_impl(self, ...):
    return f"impl_output = impl_model(*{inputs})\n"

def needs_binary_io(self):
    return False

def needs_compilation(self):
    return False

def get_special_setup_code(self):
    return """import tilelang
tilelang.cache.clear_cache()
try:
    from akg_agents.op.utils.tilelang_compile_patch import apply_tilelang_patches
    apply_tilelang_patches()
except ImportError:
    pass
"""
```

benchmark_impl 先用简单的 `time.time()` + `torch.npu.synchronize()` 传统计时即可，后续再优化 L2 cache 清除。

---

## Step 3：注册 DSL

### 3.1 注册到 `config_utils.py`

> 文件: `python/akg_agents/op/utils/config_utils.py`

**`normalize_dsl()` 第70行**，在合法 DSL 列表中添加：
```python
if dsl in ["triton_cuda", "triton_ascend", "triton-russia", "swft", "cuda_c",
            "cpp", "tilelang_npuir", "tilelang_cuda", "tilelang_ascend",
            "ascendc", "torch", "pypto"]:
    return dsl
```

**`check_dsl()` 第102行**，`valid_dsls` 列表添加 `"tilelang_ascend"`：
```python
valid_dsls = ["triton_cuda", "triton_ascend", "triton-russia", "swft", "cuda_c",
              "cpp", "tilelang_npuir", "tilelang_cuda", "tilelang_ascend",
              "ascendc", "torch", "pypto"]
```

**`VALID_CONFIGS` 第122-215行**，在所有 `torch.ascend.*` 配置的 DSL 列表末尾加 `"tilelang_ascend"`：
```python
# 例如：
"ascend910b4": ["triton_ascend", "triton-russia", "tilelang_npuir", 
                "tilelang_ascend", "ascendc", "torch", "pypto"],
```
所有 ascend910*/ascend950* 架构都要加。

### 3.2 注册到 `factory.py`

> 文件: `python/akg_agents/op/verifier/adapters/factory.py`

在 `get_dsl_adapter()` 中添加：
```python
elif dsl_lower == "tilelang_ascend":
    from .dsl.tilelang_ascend import DSLAdapterTilelangAscend
    return DSLAdapterTilelangAscend()
```

### 3.3 注册到 `kernel_verifier.py` 的 `impl_func_name` 逻辑

> 文件: `python/akg_agents/op/verifier/kernel_verifier.py` 第166-175行

在 `impl_func_name` 判断中，`tilelang_ascend` 要跟 triton 一样走 `ModelNew` 路径：
```python
if "triton_cuda" in self.dsl or "triton_ascend" in self.dsl:
    self.impl_func_name = impl_func_name or "ModelNew"
elif self.dsl == "torch":
    self.impl_func_name = impl_func_name or "ModelNew"
elif self.dsl == "tilelang_ascend":
    self.impl_func_name = impl_func_name or "ModelNew"
elif self.dsl == "ascendc":
    ...
```

---

## Step 4：创建配置文件

### 4.1 `default_tilelang_ascend_config.yaml`

> 新建: `python/akg_agents/op/config/default_tilelang_ascend_config.yaml`

```yaml
log_dir: "~/akg_agents_logs"
default_workflow: "default_workflow"
max_step: 20

docs_dir:
  designer: "op/resources/docs/sketch_docs"
  coder: "op/resources/docs/tilelang_ascend_docs"

profile_settings:
  run_times: 50
  warmup_times: 5

verify_timeout: 600
```

### 4.2 `tilelang_ascend_coderonly_config.yaml`

> 新建: `python/akg_agents/op/config/tilelang_ascend_coderonly_config.yaml`

```yaml
log_dir: "~/akg_agents_logs"
default_workflow: "coder_only_workflow"
max_step: 20

docs_dir:
  designer: "op/resources/docs/sketch_docs"
  coder: "op/resources/docs/tilelang_ascend_docs"

profile_settings:
  run_times: 50
  warmup_times: 5

verify_timeout: 600
```

---

## Step 5：编写 TileLang-Ascend kernel 测试代码

### 5.1 ReLU

> 新建: `tests/op/resources/relu_op/relu_tilelang_ascend_torch.py`

```python
import tilelang
import tilelang.language as T
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        M, N = x.shape
        block_M = min(M, 256)
        block_N = min(N, 1024)

        @tilelang.jit(out_idx=[-1])
        def relu(M, N, block_M, block_N, dtype="float16"):
            m_num = T.ceildiv(M, block_M)
            n_num = T.ceildiv(N, block_N)

            @T.prim_func
            def main(
                X: T.Tensor((M, N), dtype),
                Y: T.Tensor((M, N), dtype),
            ):
                with T.Kernel(m_num * n_num) as (cid,):
                    bx = cid // n_num
                    by = cid % n_num
                    x_ub = T.alloc_ub((block_M, block_N), dtype)
                    y_ub = T.alloc_ub((block_M, block_N), dtype)
                    T.copy(X[bx * block_M, by * block_N], x_ub)
                    T.tile.maximum(y_ub, x_ub, 0.0)
                    T.copy(y_ub, Y[bx * block_M, by * block_N])
            return main

        func = relu(M, N, block_M, block_N)
        y = torch.empty_like(x)
        func(x, y)
        return y
```

### 5.2 Linear

> 新建: `tests/op/resources/linear_op/linear_tilelang_ascend_torch.py`

```python
import tilelang
import tilelang.language as T
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        torch.manual_seed(0)
        linear = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(linear.weight.clone())
        self.bias = nn.Parameter(linear.bias.clone()) if linear.bias is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        weight = self.weight.contiguous()
        B, K = x.shape
        N = weight.shape[0]
        block_size = 16

        @tilelang.jit(out_idx=[-1])
        def linear_kernel(B, K, N, block_size, dtype="float16"):
            @T.prim_func
            def main(
                X: T.Tensor((B, K), dtype),
                W: T.Tensor((N, K), dtype),
                Bias: T.Tensor((N,), dtype),
                Y: T.Tensor((B, N), dtype),
            ):
                with T.Kernel(B * N) as (cid,):
                    bi = cid // N
                    ni = cid % N

                    acc_ub = T.alloc_local((1,), dtype)
                    T.tile.fill(acc_ub, 0.0)

                    for ki in T.serial(T.ceildiv(K, block_size)):
                        x_ub = T.alloc_ub((1, block_size), dtype)
                        w_ub = T.alloc_ub((1, block_size), dtype)
                        # 注意: 需要处理padding mask
                        cur_k = T.min(block_size, K - ki * block_size)
                        T.copy(X[bi, ki * block_size], x_ub)
                        T.copy(W[ni, ki * block_size], w_ub)
                        tmp = T.alloc_ub((1, 1), dtype)
                        T.tile.mul(tmp, x_ub, w_ub)
                        T.tile.add(acc_ub, acc_ub, tmp)

                    bias_val = T.alloc_ub((1,), dtype)
                    T.copy(Bias[ni], bias_val)
                    T.tile.add(acc_ub, acc_ub, bias_val)
                    T.copy(acc_ub, Y[bi, ni])
            return main

        bias = self.bias if self.bias is not None else torch.zeros(N, device=x.device, dtype=x.dtype)
        y = torch.empty(B, N, dtype=x.dtype, device=x.device)

        func = linear_kernel(B, K, N, block_size)
        func(x, weight, bias, y)
        return y
```

> **注意**：Linear 的 tilelang 实现较复杂（需要手动做循环和 tiling），如果暂时跑不通，优先确保 ReLU 通过。

---

## Step 6：编写测试用例

> 在 `tests/op/st/test_kernel_verifier.py` 末尾添加：

```python
@pytest.mark.level0
@pytest.mark.torch
@pytest.mark.tilelang
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.parametrize("op_name", ["relu"])
@pytest.mark.asyncio
async def test_kernel_verifier_tilelang_ascend_ascend910b4_torch(op_name):
    framework = "torch"
    dsl = "tilelang_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    config = load_config(dsl, backend=backend)

    # 读取框架实现代码
    op_task_file = f"./tests/op/resources/{op_name}_op/{op_name}_{framework}.py"
    with open(op_task_file, "r", encoding="utf-8") as f:
        op_task_str = textwrap.dedent(f.read())

    # 读取 tilelang_ascend 实现代码
    kernel_path = f"./tests/op/resources/{op_name}_op/{op_name}_{dsl}_{framework}.py"
    with open(kernel_path, "r", encoding="utf-8") as f:
        kernel_code = f.read()

    log_dir = create_log_dir(f'{op_name}_{framework}_{backend}_{arch}_{dsl}_test')

    await register_local_worker([device_id], backend=backend, arch=arch)
    worker = await get_worker_manager().select(backend=backend, arch=arch)
    if not worker:
        raise RuntimeError(f"No available worker for backend={backend}, arch={arch}")

    impl_func_name = "ModelNew"
    verifier = KernelVerifier(
        op_name=op_name,
        framework_code=op_task_str,
        framework=framework,
        dsl=dsl,
        backend=backend,
        arch=arch,
        impl_func_name=impl_func_name,
        config=config,
        worker=worker
    )
    task_info = {"coder_code": kernel_code}
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"验证失败: {error_log}"
```

---

## Step 7：运行测试

```bash
cd /home/yyz/akg/akg_agents
source env.sh

# 确保 tilelang 环境也在
cd /home/yyz && source env_yyz.sh
cd /home/yyz/tilelang-ascend && source set_env.sh
cd /home/yyz/akg/akg_agents

# 跑单个测试
python -m pytest tests/op/st/test_kernel_verifier.py::test_kernel_verifier_tilelang_ascend_ascend910b4_torch -x -v -s
```

---

## 验证流程（KernelVerifier 内部做了什么）

对照测试代码中的调用链：

```
KernelVerifier(op_name="relu", dsl="tilelang_ascend", ...)
  └── verifier.run(task_info={"coder_code": relu_kernel_code}, device_id=0)
        ├── gen_verify_project(impl_code, verify_dir, device_id)
        │     ├── 写入 framework: relu_torch.py
        │     ├── 写入 impl:       relu_tilelang_ascend_impl.py
        │     │     └── = dsl_adapter.get_import_statements() + impl_code
        │     └── 生成 verify_relu.py (Jinja2模板)
        │           ├── from relu_torch import FrameworkModel  ← get_framework_import
        │           ├── from relu_tilelang_ascend_impl import ModelNew ← get_impl_import
        │           ├── impl_model = ModelNew(*init_params)     ← create_impl_module
        │           └── impl_output = impl_model(*inputs)       ← call_impl
        └── run_verify() → worker 执行 python verify_relu.py
```

---

## 调试技巧

如果验证失败：

```bash
# 1. 查看生成的验证脚本
ls ~/akg_agents_logs/Task_*/relu/*/verify_relu.py

# 2. 手动跑验证脚本看错误
cd ~/akg_agents_logs/Task_*/relu/Iteration*_verify/
python verify_relu.py

# 3. 在 tilelang kernel 中加 printf
#    在 @T.prim_func 内部:
#    T.printf("M=%d, N=%d, bx=%d, by=%d\n", M, N, bx, by)
```

---

## 后续工作（按优先级）

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🟠 P1 | 补充 Prompt 模板的 tilelang 代码骨架 | `user_prompt.j2` 和 `codegen.j2` |
| 🟠 P1 | 创建 Skills 目录 `skills/tilelang-ascend/` | 供 LLM Agent 学习 |
| 🟡 P2 | 完善 benchmark_impl L2 cache 清除 | 走 `dsl="tilelang_ascend"` 专用路径 |
| 🟡 P2 | 补充 API 文档 `docs/tilelang_ascend_docs/api/` | 细化每个 T.tile.xxx 的用法 |
| 🟢 P3 | Linear 算子的 kernel 代码打磨 | 当前手动 tiling 写法需要验证 |
