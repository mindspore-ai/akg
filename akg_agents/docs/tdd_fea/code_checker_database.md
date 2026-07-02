# CodeChecker Database

## 1. 背景

Verifier 失败成本高：需要启动验证脚本、占用设备，并且很多失败其实在进入设备前就能发现。Triton Coder 常见的早期错误包括：

- Python 语法错误或生成了空代码。
- import 不可用。
- 生成结果中混入中文解释文本。
- 明明要求 Triton，却在 `forward()` 中调用 PyTorch 高层计算 API 作弊。
- 使用 Triton API 时存在明显签名或高置信语义风险。

CodeChecker 的目标是在 Coder 后、Verifier 前做快速静态检查，把确定性错误尽早送回 Coder 修复；同时提供非阻塞 Triton 诊断，把高置信风险写入下一轮 prompt，但不因为诊断猜测直接阻断 verifier。

## 2. 总览

启用 CodeChecker 后，生成验证路径变为：

```text
coder -> code_checker -> verifier
  ^          |
  |          +-- blocking check failed
  +--------------------------------
```

CodeChecker 分两层：

- blocking 静态检查：失败时直接回到 Coder。
- non-blocking Triton diagnostics：只写入 `code_diagnostic_*` 字段，不改变路由。

这种设计让确定性错误快速修复，同时避免诊断误报导致本来可以通过的代码不能进入 verifier。

## 3. 使用细节

workflow 中默认开启：

```yaml
enable_code_checker: true
```

如果配置缺省，当前 MathIR workflows 默认按开启处理。可显式关闭：

```yaml
enable_code_checker: false
```

非阻塞诊断配置：

```yaml
code_checker:
  base_checkers: all
  triton_checkers: all

code_diagnostic_checker:
  enabled: true
  only_errors: true
  dedup: true
```

字段含义：

- `code_checker.base_checkers`：blocking base checker pipeline。可写 `all`，也可写列表：`empty_code`、`python_syntax`、`py_compile`、`import_availability`、`stray_chinese`、`triton_dsl_compliance`。
- `code_checker.triton_checkers`：non-blocking Triton diagnostic pipeline。可写 `all`，也可写列表：`api_signature`、`high_confidence_semantics`。写空列表 `[]` 可关闭此组 pipeline。
- `enabled`：是否运行非阻塞诊断检查。
- `only_errors`：只保留 error 级别诊断。
- `dedup`：对诊断结果去重。

Coder 会收到这些信息：

- `code_check_errors`：blocking 静态检查错误。
- `code_diagnostic_errors`：非阻塞 Triton 诊断信息。
- `verifier_error`：Verifier runtime/correctness/performance 错误。

## 4. 技术细节

主要实现文件：

- `python/akg_agents/op/utils/code_checker/`
- `python/akg_agents/op/utils/code_checker/base.py`
- `python/akg_agents/op/utils/code_checker/registry.py`
- `python/akg_agents/op/utils/code_checker/base_checkers/`
- `python/akg_agents/op/utils/code_checker/triton_checkers/`
- `python/akg_agents/op/langgraph_op/nodes.py`

blocking 检查流程：

1. 空代码检查：生成结果为空或只有空白时直接失败。
2. `ast.parse` Python 语法检查：捕获括号、缩进、非法字符、markdown fence/XML tag 混入等语法错误。
3. `py_compile` 编译检查：在语法通过后捕获编译期错误。
4. import 可用性检查：提取 `import` / `from ... import ...` 的顶层模块，用 `importlib.util.find_spec` 判断环境中是否存在。
5. 中文文本混入检测：通过 `tokenize` 跳过注释和字符串，只检查真实代码 token 中连续 3 个以上汉字。
6. DSL 合规性检查：Triton DSL 下必须定义 `@triton.jit` kernel，必须通过 `kernel[grid](...)` 启动，并禁止在 `forward()` 中用 PyTorch 高层核心计算 API 替代 kernel。

当前 checker 文件按职责拆分：

- `op/utils/code_checker/__init__.py`：兼容旧 import 路径，导出 `CodeChecker`。
- `op/utils/code_checker/code_checker.py`：总控 orchestrator，按 YAML 选择结果构造 base/triton 两阶段 checker list。
- `op/utils/code_checker/registry.py`：全局 checker 注册表，YAML 中的 checker 名称通过这里解析。
- `op/utils/code_checker/base.py`：公共基类、dataclass、bool/list 配置解析、错误格式化工具；`BlockingCodeChecker` 和 `TritonDiagnosticChecker` 分别约束两类 checker。
- `op/utils/code_checker/base_checkers/`：blocking base checks，失败时 `CodeChecker.check()` 返回 `passed=False`，workflow 回到 Coder。
- `op/utils/code_checker/base_checkers/runner.py`：base checker 执行顺序和 blocking short-circuit。
- `op/utils/code_checker/base_checkers/empty_code_checker.py`：空代码检查。
- `op/utils/code_checker/base_checkers/python_syntax_checker.py`：`ast.parse` 语法检查。
- `op/utils/code_checker/base_checkers/py_compile_checker.py`：`py_compile` 编译检查。
- `op/utils/code_checker/base_checkers/import_checker.py`：import 可用性检查。
- `op/utils/code_checker/base_checkers/stray_chinese_checker.py`：未注释中文文本检查。
- `op/utils/code_checker/base_checkers/triton_dsl_compliance_checker.py`：Triton DSL 合规性检查；虽然面向 Triton，但属于 blocking base check。
- `op/utils/code_checker/triton_checkers/`：non-blocking Triton diagnostics，发现问题时只写入 `last_diagnostic_*`，不阻止 verifier。
- `op/utils/code_checker/triton_checkers/runner.py`：Triton diagnostic 执行顺序、过滤和 non-blocking 结果聚合。
- `op/utils/code_checker/triton_checkers/api_signature_checker.py`：Triton API signature diagnostic checker。
- `op/utils/code_checker/triton_checkers/high_confidence_semantics_checker.py`：高置信 Triton 语义 diagnostic checker。

非阻塞 Triton 诊断流程：

1. registry 注册可选 checker。
2. `code_checker.triton_checkers` 从 YAML 选择要运行的 checker list。
3. runner 执行 API signature 和 high-confidence semantics 等已选检查。
4. 结果写入 `CodeChecker.last_diagnostic_*`。
5. `code_checker_node` 将结果写回 state 的 `code_diagnostic_*` 字段。

日志和计数：

- `code_checker` 使用 `write_record` 写日志，不消耗 `step_count`。
- blocking 失败时，`create_code_checker_router` 路由回 Coder。
- diagnostics 不影响路由，只作为下一轮 Coder 的 repair context。
