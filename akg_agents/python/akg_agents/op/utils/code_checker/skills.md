# CodeChecker Checker Skill Guide

This directory owns all CodeChecker checks. To add a new checker, implement it as
an OOP checker class, register it in the matching checker group, and select it
from YAML by its canonical `name`.

## Directory Roles

- `code_checker.py`: top-level orchestrator. It builds the selected base and
  Triton checker lists from YAML and runs the two-stage pipeline.
- `registry.py`: global checker registry. YAML names are resolved here.
- `base.py`: shared checker base classes, error/diagnostic dataclasses, and
  formatting helpers.
- `base_checkers/`: blocking checks. Any emitted error makes
  `CodeChecker.check()` return `passed=False`, so workflow routes back to Coder.
- `triton_checkers/`: non-blocking Triton diagnostics. Issues are recorded in
  `last_diagnostic_*`; verifier still runs when base checks pass.

## Add A Blocking Base Checker

Use this for syntax, compile, import, DSL compliance, or any rule that should
stop verifier and return to Coder.

1. Create a file under `base_checkers/`, for example
   `base_checkers/my_checker.py`.

```python
from typing import Dict, List

from akg_agents.op.utils.code_checker.base import BlockingCodeChecker


class MyChecker(BlockingCodeChecker):
    name = "my_checker"

    def check(self, code: str) -> List[Dict]:
        errors = []
        if "bad_pattern" in code:
            errors.append(
                {
                    "line": 0,
                    "error_type": "my_checker_error",
                    "detail": "Found bad_pattern in generated code.",
                    "suggestion": "Remove bad_pattern or replace it with valid code.",
                    "code_snippet": "",
                }
            )
        return errors
```

2. Register it in `base_checkers/__init__.py`.

```python
from akg_agents.op.utils.code_checker.base_checkers.my_checker import MyChecker

registry.register(
    CheckerSpec(
        name=MyChecker.name,
        group="base",
        factory=MyChecker,
    )
)
```

3. Enable it in YAML.

```yaml
code_checker:
  base_checkers: all
```

or select it explicitly:

```yaml
code_checker:
  base_checkers:
    - empty_code
    - python_syntax
    - my_checker
```

## Add A Non-Blocking Triton Checker

Use this for Triton API or high-confidence semantic diagnostics that should help
repair but should not stop verifier.

1. Create a file under `triton_checkers/`, for example
   `triton_checkers/my_triton_checker.py`.

```python
from typing import List

from akg_agents.op.utils.code_checker.base import (
    CheckContext,
    Issue,
    Location,
    TritonDiagnosticChecker,
)


class MyTritonChecker(TritonDiagnosticChecker):
    name = "my_triton_checker"
    checker_id = "my_triton_checker"
    rule_id = "MY_TRITON_RULE"

    def run(self, code: str, ctx: CheckContext) -> List[Issue]:
        if not (ctx.dsl or "").lower().startswith("triton"):
            return []

        if "bad_triton_pattern" not in code:
            return []

        return [
            Issue(
                severity="ERROR",
                rule_id=self.rule_id,
                title="Bad Triton pattern",
                message="bad_triton_pattern is not supported.",
                location=Location(lineno=0, col=0),
                hint="Use a supported Triton pattern.",
                tags={"triton", "kernel"},
            )
        ]
```

2. Register it in `triton_checkers/__init__.py`.

```python
from akg_agents.op.utils.code_checker.triton_checkers.my_triton_checker import (
    MyTritonChecker,
)

registry.register(
    CheckerSpec(
        name=MyTritonChecker.name,
        group="triton",
        factory=MyTritonChecker,
    )
)
```

If the checker should run when `triton_checkers: all`, also add it to
`default_triton_checkers()` in the desired order.

3. Enable it in YAML.

```yaml
code_checker:
  triton_checkers: all

code_diagnostic_checker:
  enabled: true
  only_errors: true
  dedup: true
```

or select it explicitly:

```yaml
code_checker:
  triton_checkers:
    - api_signature
    - my_triton_checker
```

## YAML Selection Rules

- `all`: run every registered checker in registration order.
- A list: run only canonical checker names in registration order.
- `[]`, `false`, `none`, `off`, or `disabled`: run no checker in that group.
- Unknown names are ignored with a warning.
- Aliases are intentionally not supported; use each checker class `name`.

## Behavior Rules

- Do not edit `code_checker.py` for a normal new checker. The orchestrator should
  stay generic.
- Put blocking behavior in `base_checkers/`; put advisory Triton diagnostics in
  `triton_checkers/`.
- Base checker errors must use the CodeChecker error dict fields:
  `line`, `error_type`, `detail`, `suggestion`, `code_snippet`.
- Triton checker issues must return `Issue` objects.
- Keep checker names stable, because YAML config depends on them.
- Keep each checker focused on one responsibility. Shared helpers can live next
  to the checker group when more than one checker needs them.

## Validation

After adding a checker, run:

```bash
PYTHONPATH=akg_agents/python python -m py_compile \
  akg_agents/python/akg_agents/op/utils/code_checker/*.py \
  akg_agents/python/akg_agents/op/utils/code_checker/base_checkers/*.py \
  akg_agents/python/akg_agents/op/utils/code_checker/triton_checkers/*.py

source akg_agents/env.sh >/dev/null 2>&1 && \
PYTHONPATH=akg_agents/python python -m pytest -q \
  akg_agents/tests/op/ut/test_code_checker.py
```
