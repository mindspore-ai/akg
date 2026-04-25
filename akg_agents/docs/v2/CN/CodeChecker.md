[English](../CodeChecker.md)

# CodeChecker

## 1. 用途

在生成的 kernel 代码送入 `KernelVerifier` 之前执行的静态检查器。
使用 `ast.parse`、`py_compile`、`importlib` 与正则；不调用 LLM，不在
设备上执行代码。拦截会导致验证失败的错误（语法错、模块无法解析、
代码中存在非标识符文本），并检查 runtime verifier 无法识别的工程
规约（kernel 被定义且启动；`forward()` 不调用 torch 计算 API；
`@triton.autotune` 声明 `restore_value`）。

## 2. 调用方

所有生成 kernel 代码的 workflow 均会调用：

| 调用方 | 文件 |
|---|---|
| `default_workflow` / `default_workflow_v2` | `op/workflows/default_workflow*.py` |
| `coder_only_workflow` | `op/workflows/coder_only_workflow.py` |
| `kernelgen_only_workflow` | `op/workflows/kernelgen_only_workflow.py` |
| `autoresearch_workflow` | `op/workflows/autoresearch_workflow.py` |
| AutoResearch agent 工具 | `op/autoresearch/agent/tools.py` |

调用方构造 `CodeChecker(backend, dsl)` 并 `await checker.check(code)`。
检查失败时调用方将 `error_message` 传给 coder / KernelGen 作为下一次
重试的输入。

## 3. 流水线

六步按序执行。第 1、2 步出错时，第 3–6 步不再运行。

| # | 步骤 | 实现 | 拦截的情况 |
|---|---|---|---|
| 1 | 语法 | `ast.parse` | 括号不匹配、全角标点、缩进错误 |
| 2 | 编译 | `py_compile.compile(doraise=True)` | `ast.parse` 不抛出但编译层能发现的错误（重复关键字参数、SyntaxWarning 升级为 error） |
| 3 | Import | 对每个 import 顶层模块调用 `importlib.util.find_spec` | 模块名在当前环境下无法解析 |
| 4 | 非标识符文本 | `tokenize` → 对代码 token 跑正则（注释、字符串除外） | 连续 ≥ `min_run` 个落在配置 Unicode 区间内的字符 |
| 5 | DSL 合规 | 对整个模块与 kernel 类体做 AST walk | 未定义 `@<triton>.<decorator>` kernel；kernel 定义了但未通过 `kernel[grid](...)` 启动；`forward()` 调用 `torch_compute_ops_hard` 中的方法；`forward()` 调用 `torch_compute_ops_soft` 中的方法且无 kernel 启动 |
| 6 | Autotune | 正则 + 括号匹配 | `@<triton_module_name>.<decorator_attr>(...)` 缺 `<required_kwarg>=` |

第 5、6 步仅当 `dsl.startswith(dsl_compliance_prefix)` 时运行（默认
前缀 `triton`）。

## 4. 策略（`op/config/code_checker.yaml`）

检查器使用的关键词集合与标识符定义在
[`op/config/code_checker.yaml`](../../../python/akg_agents/op/config/code_checker.yaml)
中。Python 模块无回退默认值。key 缺失或类型错误在模块 import 时
抛出 `KeyError` 或 `re.error`。

| Key | 类型 | 含义 |
|---|---|---|
| `triton_decorators` | list[str] | `<triton_module_name>` 命名空间下视作 kernel 装饰器的属性名 |
| `torch_call_prefixes` | list[str] | `forward()` 中被扫描的顶层调用前缀 |
| `torch_compute_ops_hard` | list[str] | 无论 kernel 是否启动都拒绝出现在 `forward()` 中的方法名 |
| `torch_compute_ops_soft` | list[str] | 仅在 kernel 未启动时拒绝出现在 `forward()` 中的方法名 |
| `kernel_class_name` | str | 第 5 步 AST walker 搜索的类名 |
| `kernel_forward_method` | str | 上述类中被扫描的方法名 |
| `triton_module_name` | str | Triton 装饰器匹配的顶层命名空间 |
| `dsl_compliance_prefix` | str | 激活第 5–6 步的 DSL 字符串前缀 |
| `stray_text.min_run` | int | 触发标记的最小连续字符数 |
| `stray_text.unicode_ranges` | list of `[lo, hi]` | 第 4 步扫描的码点区间 |
| `autotune.decorator_attr` | str | `<triton_module_name>` 下触发第 6 步的装饰器属性名 |
| `autotune.required_kwarg` | str | 装饰器调用中必须出现的关键字参数名 |

YAML 在模块 import 时通过
`importlib.resources.files("akg_agents.op.config")` 一次性读入，作为
`package_data` 打包（见 `setup.py` 的 `**/*.yaml`）。

## 5. 定制

所有规则改动均为 YAML 编辑，无需修改 Python：

- 在 `torch_compute_ops_hard` 与 `torch_compute_ops_soft` 之间移动
  算子以调整其被拒的条件。发布策略将 `layer_norm`、`batch_norm`、
  pooling、`interpolate`、`cumsum`、`cumprod` 放入 soft list，以接受
  将这些算子用作 kernel 前处理的 Ascend seeds。
- 从两个列表中同时删除某算子以无条件接受。
- 向任一列表追加新的方法名。
- kernel 模块命名约定变更时，修改 `kernel_class_name` /
  `kernel_forward_method`。
- 向 `stray_text.unicode_ranges` 追加其他文字区间（平假名
  `[0x3040, 0x309f]`、片假名 `[0x30a0, 0x30ff]`、谚文
  `[0xac00, 0xd7af]`）。

## 6. 接口

```python
CodeChecker(backend: str, dsl: str, config: Optional[dict] = None)
```

`config` 参数保留仅为调用方签名兼容，CodeChecker 不读取。

```python
await checker.check(code: str) -> Tuple[bool, str, List[Dict]]
```

- `passed` —— 总结果。
- `error_message` —— 带上下文行的 markdown 报告，可直接用于下一次
  coder / KernelGen prompt。
- `errors` —— dict 列表。每项含 `line`、`error_type`、`detail`、
  `suggestion`、`code_snippet`。`error_type` 取值
  （`syntax_error`、`compile_error`、`import_error`、
  `stray_chinese_text`、`no_triton_kernel`、
  `triton_kernel_not_called`、`torch_api_instead_of_kernel`、
  `torch_api_without_kernel`、`autotune_missing_restore_value`、
  `empty_code`）被 `op/autoresearch/agent/tools.py` 与 failure
  extractor 消费，改名时需同步更新这些消费者。
