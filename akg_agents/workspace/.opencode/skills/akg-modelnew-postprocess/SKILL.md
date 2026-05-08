---
name: akg-modelnew-postprocess
description: "原地修改 Dynamic Tune case 的 impl.py。"
disable-model-invocation: true
triggers:
  - "akg modelnew postprocess"
  - "modelnew后处理"
  - "akg 动态shape 后处理"
---

# AKG ModelNew 后处理 Skill

## 输入文件

- `impl.py`
- `base.py`
- `sample.json`

## 输出要求

- 原地修改 `impl.py`

## 固定流程

1. 读取 `impl.py`、`base.py`、`sample.json`
2. 根据 `sample.json` 中的 `shape_expr` 和 `config_param_names` 修改 `ModelNew`
3. 原地修改 `impl.py`
4. 执行静态 contract 校验：

```bash
PYTHONPATH=../../../../python:../../../.. python - <<'PY'
from pathlib import Path
from akg_agents.op.dynamic_tune.cases.contract import _ModelNewContractValidator

output_path = Path("impl.py")
errors = _ModelNewContractValidator.validate_code(
    output_path.read_text(encoding="utf-8"),
    output_path=str(output_path),
)
if errors:
    raise SystemExit("contract validation failed:\n- " + "\n- ".join(errors))
PY
```

5. 校验失败时，只修改 `impl.py`，然后重新执行静态 contract 校验
6. 静态 contract 校验通过后结束任务

## 修改范围

- 只允许修改 `ModelNew.__init__`
- 只允许修改 `ModelNew.forward`
- 必须新增或保留 `ModelNew._select_config`
- 不改 kernel body
- 不改 `Model`
- 不改其它 helper 函数

## ModelNew 要求

- `ModelNew.forward(..., config=None)` 必须保留 `config=None`
- `config is None` 时调用 `self._select_config(shape_key)`
- `config is not None` 时直接使用传入的显式 `Config`
- 必须使用 `from akg_agents.op.dynamic_tune import load_deployed_selector`
- `_select_config` 内调用 `load_deployed_selector()`
- `_select_config` 调用 selector 的 `select_config(shape_key)`
- `shape_key` 按 `sample.json` 的 `shape_expr` 构造
- 读取候选参数只能使用 `config.param("...")`
- `config.param("...")` 只用于 `sample.json` 的 `config_param_names` 字段
- 不在 `config_param_names` 中的 kernel 参数保持 `impl.py` 原写法
