# Copyright 2026 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""task_config package — facade over three single-concern submodules.

Layout:

    loader            — TaskConfig dataclass + load_task_config (YAML
                        parsing). No internal deps; everyone else
                        consumes TaskConfig from here.
    metric_policy     — EvalResult, is_improvement, check_constraints,
                        format_result_summary. Pure data + arithmetic;
                        no I/O. Imported by keep_or_discard, dashboard.
    eval_client       — Local subprocess + remote HTTP transport,
                        result assembly. Depends on loader +
                        metric_policy. Local drives the static
                        `eval_kernel.py` via `eval_runner.local_eval`;
                        remote ships a `package_builder` tar.gz to a
                        worker `/api/v1/run` endpoint.
    package_builder   — task.yaml + ref + editable → tar.gz bytes,
                        for the remote transport. No deps outside loader.

This `__init__.py` re-exports only the names actually imported from
outside the package. Submodule-private helpers (operator tables,
internal result assembly) are not re-laundered through the facade —
reach into the submodule explicitly when you need them.
"""
# fmt: off
from .loader import (
    TaskConfig, load_task_config,
    REF_FILE_DEFAULT, py_stem,
)
from .metric_policy import (
    EvalOutcome, EvalResult, check_constraints, is_improvement, format_result_summary,
)
from .eval_client import run_eval
# fmt: on
