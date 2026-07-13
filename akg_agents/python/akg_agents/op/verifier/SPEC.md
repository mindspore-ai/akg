# op/verifier/ — 验证器

## 职责

验证生成的内核代码的正确性和性能。通过三类适配器（backend、dsl、framework）组合支持多平台。

## 目录结构

```
verifier/
├── kernel_verifier.py         # KernelVerifier — 验证入口，组合三类适配器
├── data_cache.py              # Verifier Data Cache — reference data / baseline 持久缓存
├── sol_verifier.py            # SOL-ExecBench 格式验证的专用生成器
├── profiler.py                # NPU/CUDA 性能采集
├── profiler_utils.py          # profile 脚本执行、msprof/nsys 解析
├── roofline_utils.py          # 通过已安装的 solar Python 包计算 roofline
├── l2_cache_clear.py          # Ascend L2 cache 清理
└── adapters/                  # 三类适配器
    ├── factory.py             # get_backend_adapter / get_dsl_adapter / get_framework_adapter
    ├── backend/               # BackendAdapter 及实现
    │   ├── base.py            #   BackendAdapter(ABC)
    │   ├── cuda.py            #   BackendAdapterCuda
    │   ├── ascend.py          #   BackendAdapterAscend
    │   └── cpu.py             #   BackendAdapterCpu
    ├── dsl/                   # DSLAdapter 及实现
    │   ├── base.py            #   DSLAdapter(ABC) — 扩展钩子在此
    │   ├── triton_cuda.py     #   DSLAdapterTritonCuda
    │   ├── triton_ascend.py   #   DSLAdapterTritonAscend
    │   ├── cpp.py             #   DSLAdapterCpp
    │   ├── cuda_c.py          #   DSLAdapterCudaC
    │   ├── ascendc.py         #   DSLAdapterAscendC
    │   ├── ascendc_catlass.py #   DSLAdapterAscendC_Catlass
    │   ├── tilelang_cuda.py   #   DSLAdapterTilelangCuda
    │   ├── tilelang_npuir.py  #   DSLAdapterTilelangNpuir
    │   ├── torch.py           #   DSLAdapterTorch
    │   ├── pypto.py           #   DSLAdapterPypto
    │   └── swft.py            #   DSLAdapterSwft
    └── framework/             # FrameworkAdapter 及实现
        ├── base.py            #   FrameworkAdapter(ABC)
        ├── torch.py           #   FrameworkAdapterTorch
        ├── mindspore.py       #   FrameworkAdapterMindSpore
        └── numpy.py           #   FrameworkAdapterNumpy
```

## 开发约定

### 新增验证适配器的标准流程

1. 确定适配器类型（backend / dsl / framework）
2. 在对应子目录创建文件，继承 `BackendAdapter(ABC)` / `DSLAdapter(ABC)` / `FrameworkAdapter(ABC)`
3. 实现 ABC 的抽象方法 + 按需 override 下文的扩展钩子
4. 在 `adapters/factory.py` 中注册名称映射
5. 在 `op/utils/config_utils.py` 的 `VALID_CONFIGS / valid_dsls` 表里追加 DSL 名

### DSLAdapter 扩展钩子（base.py）

每加一个 DSL 应该 **只** 改 adapter 文件，不要在 `kernel_verifier.py` / `local_worker.py` /
`workspace_autoresearch/scripts/scaffold.py` 等调用方加 `if self.dsl == "xxx":` 分支。
所有 per-DSL 行为通过下面的钩子表达；调用方拿到 adapter 实例后直接调用对应钩子。

**抽象方法（必须实现）**：

| 方法 | 用途 |
|---|---|
| `get_import_statements(framework)` | impl 文件顶部的 import 行 |
| `get_impl_import(op_name, impl_func_name)` | 验证脚本里 import impl 的语句 |
| `call_impl(...)` | 验证脚本里调用 impl 的代码片段 |
| `benchmark_impl(...)` | profile 模板里 benchmark 的代码片段 |

**类属性（pure constant，override 用 `= value`，不要写成 `def f(self): return value`）**：

| 属性 | 默认 | 含义 |
|---|---|---|
| `needs_binary_io` | `False` | 是否需要二进制 I/O 文件中转（swft = True） |
| `static_check_via_python_ast` | `True` | LLM 提交的源码是否合法 Python（cpp / cuda_c / swft = False，`CodeChecker` 跳过 AST 层） |
| `profile_via_python_script` | `False` | LocalWorker dispatch：True → 跑 Python 脚本读 JSON；False → 走 msprof / nsys（triton_* / pypto / catlass = True） |
| `benchmark_requires_l2_clear` | `True` | base benchmark 模板是否每次清 L2（catlass = False） |
| `impl_func_name_template` | `"{op_name}_{dsl}_{framework}"` | 默认 impl 函数名模板（ModelNew 类 DSL = `"ModelNew"`，AscendC = `"{op_name}_kernel"`） |
| `kernel_arg_is_directory` | `False` | DSL 的 kernel handoff 是不是目录（ascendc_catlass / ascendc = True：目录里有 sibling Python wrapper + 项目子树） |
| `kernel_project_dir_name` | `None` | `kernel_arg_is_directory=True` 时的项目子目录名（如 `"catlass_op"` / `"ascendc_op"`） |
| `kernel_project_files` | `[]` | 构成 DSL kernel 项目的文件清单（相对 Python wrapper 同级目录），驱动层（如 WA scaffold）据此决定要拷哪些 / 设哪些可编辑 |

**可选钩子（默认 no-op，按需 override）**：

| 钩子 | 调用方 | 用途 |
|---|---|---|
| `materialize_impl(impl_code, verify_dir, op_name, framework, dsl_name, task_info, config)` | `KernelVerifier.gen_verify_project` | 把 LLM 生成的代码落到 verify_dir。默认写 `<op>_<dsl>_impl.py` 并 prepend imports；catlass 写 kernel.py + 拷 catlass_op 树 |
| `expected_artifacts(verify_dir, op_name, framework, bench_type, dsl_filename_hint)` | `KernelVerifier._verify_impl_artifacts_ready` | 列 verify_dir 必备产物路径；默认是 framework 文件 + impl 文件 |
| `prepare_config(config, task_info)` | `KernelVerifier.run / run_profile`、调用 `get_special_setup_code` 之前 | 每轮跑前的 config 副作用（resolve CATLASS_ROOT 等） |
| `get_special_setup_code(framework)` | impl 文件顶部 | 注入一次性 setup 片段（tilelang clear_cache、catlass cmake build 等）；arch / catlass_root 等不要进签名，从 `prepare_config` 缓存到 self 上读 |
| `get_runtime_env_override_code(**kwargs)` | impl 文件顶部 | 注入运行时 env 覆盖（pypto 运行模式 / 调试位）；默认空字符串 |
| `post_iteration_cleanup(verify_dir)` | WA `akg_eval._eval_async` finally 块 | 每轮结束后清理短命产物（catlass 删 `catlass_op/build`） |
| `read_kernel_source(kernel_arg, op_name=None)` | WA scaffold | 把 kernel handoff 路径解析成 `(source_code, project_dir_or_None)`；默认按文件读，catlass 按目录读 + 同级 `kernel.py` / `<op>_kernel.py` 兜底 |
| `materialize_project_tree(dst_dir, project_src, project_dir_name=None)` | WA scaffold | 把项目子树拷到 dst_dir 并做 DSL 特定修补（catlass 拷项目目录 + patch CMakeLists） |

### 调用方约定

| 调用方 | 通过 adapter 拿到的能力 |
|---|---|
| `KernelVerifier.gen_verify_project` | `materialize_impl` + `get_import_statements` + `get_impl_import` + `get_special_setup_code` + `get_runtime_env_override_code` + `impl_func_name_template` + `needs_binary_io` |
| `KernelVerifier.run / run_profile` | `prepare_config` + `expected_artifacts` + `benchmark_requires_l2_clear` |
| `LocalWorker.profile`（`core/worker/local_worker.py`） | `profile_via_python_script` → 决定走 Python-script 路径还是 msprof / nsys |
| `CodeChecker.check`（`op/utils/code_checker.py`） | `static_check_via_python_ast` → 是否跑 AST 层检查 |
| WA `scripts/scaffold.py` | `read_kernel_source` + `materialize_project_tree` + `kernel_project_files`（WA 据此派生 task.yaml `editable_files`；"editable" 这层语义是 WA 策略，不放 adapter 上） |
| WA `scripts/batch/manifest.py` | `kernel_arg_is_directory` + `kernel_project_dir_name`（决定 batch 单文件 vs 多文件解析路径） |
| WA `scripts/utils/akg_eval.py` | `post_iteration_cleanup` |

### KernelVerifier 核心逻辑

`KernelVerifier` 通过 `get_*_adapter` 工厂方法获取三个适配器实例（DSL adapter 在
`__init__` 里 cache 到 `self.dsl_adapter`，因为 `prepare_config` 会在 adapter 实例
上 stash state），然后组合生成验证脚本（Jinja2 模板）、CMake 配置等，最终执行验证和 profiling。

验证包和回传 artifact 都是跨进程/HTTP 边界的数据：LocalWorker 解包时必须拒绝
绝对路径、`..` 越界、链接和设备节点；`sync_artifacts_to_directory` 必须在写入前
确认 realpath 仍位于当前 verify_dir，不能信任远端返回的相对路径。

### Verifier Data Cache

- 仅作用于 `KernelBench` 风格验证链路
- reference data cache：复用 `generate_reference_data(save_inputs=True)` 产出的 `.pt`
- baseline cache：复用 `base_profile_result.json` / `avg_time_us`
- 默认关闭；开启后在 `~/.akg/verifier_data_cache/` 下持久化
- 命中 reference data 时，验证脚本改为直接复用 inputs / outputs，不再重复执行 framework baseline
- reference data cache 仅覆盖静态 shape；动态 shape 自动跳过，避免误复用单组输入
- reference data 命中后会校验 `.pt` payload，损坏或缺少可复用字段时删除旧缓存并重新生成
- 命中 baseline cache 时，profile 直接注入 `override_base_time_us` 并跳过 base profile 脚本
- cache key 默认包含 `task_id`；配置了 `data_cache.cache_key_id` 时使用该稳定身份，以支持同一工作流内多个 verifier task 复用 cache
- baseline cache key 还必须包含 DSL，避免同一 framework/backend/arch 下不同计时路径相互污染

### Roofline 集成

- roofline 通过 `roofline_utils.py` 直接调用已安装的 `solar` Python API（`graph/einsum/analysis/perf`）
- AKG **不依赖**本地 `SOLAR` 工作树路径；运行时只要求 `import solar` 成功
- 原先不在 Solar 正式包里的接入逻辑（如 solbench wrapper / Ascend arch config）由 AKG 自己维护
- **不要**修改 `SOLAR` 仓库源码或对其打 patch
- roofline 失败只能降级为“无 roofline 数据”，**不能**影响原有 correctness / profile 主流程

## Autotune 双模式验证

`KernelVerifier` 对 Triton autotune 代码支持两种验证模式：

- **直接验证模式**（默认）：autotune 代码和普通代码一样，一次性跑完整代码验证。
- **逐 config 验证模式**：逐个 config 单独验证，全部通过后再跑一次完整代码回归验证。如果逐 config 通过但完整代码失败，日志会提示添加 `restore_value`。

逐 config 验证模式有两种开启方式（任一生效）：
1. 环境变量：`AKG_VERIFY_PER_CONFIG=1`
2. triton_config YAML 配置：`verify_per_config: true`（默认 `false`）

两种模式均要求 `@triton.autotune` 必须包含 `restore_value` 参数（由 `CodeChecker` 静态检查保障）。

## 不做什么

- **不要**在适配器中实现 Agent/Workflow 逻辑
- **不要**硬编码后端/DSL 特定行为到 `kernel_verifier.py` / `local_worker.py` /
  `workspace_autoresearch/scripts/scaffold.py` ——通过适配器扩展钩子。新增 `if
  self.dsl == "xxx":` 分支审查时会被打回；正确做法是在 `DSLAdapter` 里加扩展钩子
  / 类属性，调用方只 query adapter
- **不要**把纯常量写成 `def f(self): return value` 的方法。pure constant 应写成类
  属性（`flag: bool = False`），方法体只留给真有副作用 / 计算的钩子（如
  `prepare_config`, `materialize_impl`, `post_iteration_cleanup`）
- **不要**在 base `DSLAdapter` 的方法签名里塞某个 DSL 私有的参数（如 `arch`,
  `catlass_root`）。这是"接口泄漏"——应让 adapter 在 `prepare_config` 里从
  config 读出来缓存到 `self.*`，`get_special_setup_code(framework)` 等签名保持
  跨 DSL 一致
