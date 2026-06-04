[中文版](./CN/CodeChecker.md)

# CodeChecker

## 1. Purpose

Static checker executed on generated kernel code before
`KernelVerifier`. Uses `ast.parse`, `py_compile`, `importlib`, and
regex; does not invoke an LLM or run code on a device. Catches errors
that would fail verification (syntax, unresolvable imports,
non-identifier text in code) and enforces project conventions that the
runtime verifier does not detect (kernel defined and launched;
`forward()` does not call torch compute APIs; `@triton.autotune`
declares `restore_value`).

## 2. Callers

Invoked by every workflow that generates kernel code:

| Caller | File |
|---|---|
| `default_workflow` / `default_workflow_v2` | `op/workflows/default_workflow*.py` |
| `coder_only_workflow` | `op/workflows/coder_only_workflow.py` |
| `kernelgen_only_workflow` | `op/workflows/kernelgen_only_workflow.py` |
| `autoresearch_workflow` | `op/workflows/autoresearch_workflow.py` |
| AutoResearch agent tools | `op/autoresearch/agent/tools.py` |

Each caller constructs `CodeChecker(backend, dsl)` and awaits
`checker.check(code)`. On failure the caller forwards `error_message`
to the coder / KernelGen for the next retry.

## 3. Pipeline

Six stages, run in order. Stages 3–6 are skipped when stage 1 or 2
reports an error.

| # | Stage | Implementation | Rejected cases |
|---|---|---|---|
| 1 | Syntax | `ast.parse` | unmatched brackets, fullwidth punctuation, indentation errors |
| 2 | Compile | `py_compile.compile(doraise=True)` | errors `ast.parse` does not raise (duplicate kwargs, SyntaxWarning promoted to error) |
| 3 | Imports | `importlib.util.find_spec` on each top-level import target | module name not resolvable in the current environment |
| 4 | Non-identifier text | `tokenize` → regex over code tokens (comments and strings excluded) | runs of ≥ `min_run` consecutive characters from a configured Unicode range |
| 5 | DSL compliance | AST walk over the module and the kernel class body | no `@<triton>.<decorator>` kernel defined; kernel defined but never launched via `kernel[grid](...)`; `forward()` calls a method in `torch_compute_ops_hard`; `forward()` calls a method in `torch_compute_ops_soft` while no kernel is launched |
| 6 | Autotune | regex + paren matching | `@<triton_module_name>.<decorator_attr>(...)` missing `<required_kwarg>=` |

Stages 5 and 6 run only when `dsl.startswith(dsl_compliance_prefix)`
(default prefix `triton`).

## 4. Policy (`op/config/code_checker.yaml`)

The checker's keyword sets and identifiers are defined in
[`op/config/code_checker.yaml`](../../python/akg_agents/op/config/code_checker.yaml).
The Python module has no fallback defaults. A missing or ill-typed
key raises `KeyError` or `re.error` at module import.

| Key | Type | Meaning |
|---|---|---|
| `triton_decorators` | list[str] | attribute names under `<triton_module_name>` recognized as kernel decorators |
| `torch_call_prefixes` | list[str] | local aliases or dotted torch namespaces whose `.method(...)` calls are scanned in `forward()` |
| `torch_compute_ops_hard` | list[str] | methods rejected in `forward()` regardless of whether a kernel is launched |
| `torch_compute_ops_soft` | list[str] | methods rejected in `forward()` only when no kernel is launched |
| `kernel_class_name` | str | class name the AST walker searches for in stage 5 |
| `kernel_forward_method` | str | method within that class whose body is scanned |
| `triton_module_name` | str | top-level namespace for the Triton decorator match |
| `dsl_compliance_prefix` | str | DSL-string prefix that activates stages 5–6 |
| `stray_text.min_run` | int | minimum consecutive characters required to flag |
| `stray_text.unicode_ranges` | list of `[lo, hi]` | code-point ranges scanned in stage 4 |
| `autotune.decorator_attr` | str | attribute name off `<triton_module_name>` that triggers stage 6 |
| `autotune.required_kwarg` | str | keyword argument that must appear in the decorator call |

The YAML is read once at module import via
`importlib.resources.files("akg_agents.op.config")` and is packaged as
`package_data` (see `setup.py` `**/*.yaml`).

## 5. Customizing

All rule changes are YAML edits; no Python changes required.

- Move an op between `torch_compute_ops_hard` and
  `torch_compute_ops_soft` to change when it is rejected. The shipped
  policy places `layer_norm`, `batch_norm`, pooling, `interpolate`,
  `cumsum`, `cumprod` in the soft list so Ascend seeds using them as
  kernel pre-processing are accepted.
- Remove an op from both lists to accept it unconditionally.
- Append method names to either list.
- Append local aliases or dotted namespaces to `torch_call_prefixes`;
  examples include `F`, `torch.nn.functional`, `nn.functional`,
  `torch.linalg`, `torch.fft`, `torch.special`, and `torch.sparse`.
- Update `kernel_class_name` / `kernel_forward_method` when the kernel
  module convention changes.
- Extend `stray_text.unicode_ranges` with additional script ranges
  (Hiragana `[0x3040, 0x309f]`, Katakana `[0x30a0, 0x30ff]`,
  Hangul `[0xac00, 0xd7af]`).

## 6. Interface

```python
CodeChecker(backend: str, dsl: str, config: Optional[dict] = None)
```

The `config` parameter is retained for call-site signature
compatibility; CodeChecker does not read it.

```python
checker.check(code: str) -> Tuple[bool, str, List[Dict]]
```

- `passed` — overall result.
- `error_message` — markdown-formatted report with context lines,
  suitable for insertion into the next coder / KernelGen prompt.
- `errors` — list of dicts. Each entry has `line`, `error_type`,
  `detail`, `suggestion`, `code_snippet`. The `error_type` values
  (`syntax_error`, `compile_error`, `import_error`,
  `stray_chinese_text`, `no_triton_kernel`,
  `triton_kernel_not_called`, `torch_api_instead_of_kernel`,
  `torch_api_without_kernel`, `autotune_missing_restore_value`,
  `empty_code`) are referenced by `op/autoresearch/agent/tools.py`
  and the failure extractor; rename only with the consumers updated.
