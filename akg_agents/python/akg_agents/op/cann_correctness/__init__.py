# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""CANN-Bench evaluation standard — single, isolated owner.

Everything cannbench-specific (MERE/MARE precision, per-op thresholds/registries,
FP64 dual reference, the cann verify/profile project generators and their
templates, task loading) lives in this package. akg core reaches it only through
the exports below; the one switch is the boolean ``DSLAdapter.uses_cannbench_precision``
(True -> cannbench precision path; False/default -> framework precision).

Import order note: constants/helpers and the lightweight core/task_loader are
defined before ``verifier`` is pulled, because verifier imports the adapter
factory which imports the DSL adapters — those look up ``CORE_PY_PATH`` here. See
``get_aux_verify_files`` in adapters/dsl/base.py (local import, cycle-proof)."""

import os
import shutil

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

# core.py is staged into a verify/profile dir as ``cann_correctness.py`` — the
# name the generated scaffold imports. Single owner of that source path.
CORE_PY_PATH = os.path.join(_PKG_DIR, "core.py")
TEMPLATES_DIR = os.path.join(_PKG_DIR, "templates")


def stage_core_into(dest_dir: str) -> str:
    """Copy core.py into ``dest_dir`` as cann_correctness.py; return the path."""
    dst = os.path.join(dest_dir, "cann_correctness.py")
    shutil.copy2(CORE_PY_PATH, dst)
    return dst


from .core import (  # noqa: E402
    compare_tensors,
    assert_outputs,
    dual_reference,
    validate_index_output,
    set_seed,
)
from .task_loader import (  # noqa: E402
    is_cann_task_dir,
    load_cann_task_source,
    inject_cann_into_config,
    get_cann_task_desc_for_prompt,
    load_cann_task_for_runner,
    load_cann_proto,
    load_cann_golden,
    load_cann_desc,
)
from .verify_snippets import (  # noqa: E402
    reference_call_snippet,
    compare_snippet,
)
from .verifier import (  # noqa: E402
    generate_cann_verify_project,
    generate_cann_profile_project,
    CANN_BENCH_SRC_DIR,
)

__all__ = [
    "CORE_PY_PATH",
    "TEMPLATES_DIR",
    "stage_core_into",
    "compare_tensors",
    "assert_outputs",
    "dual_reference",
    "validate_index_output",
    "set_seed",
    "reference_call_snippet",
    "compare_snippet",
    "is_cann_task_dir",
    "load_cann_task_source",
    "inject_cann_into_config",
    "get_cann_task_desc_for_prompt",
    "load_cann_task_for_runner",
    "load_cann_proto",
    "load_cann_golden",
    "load_cann_desc",
    "generate_cann_verify_project",
    "generate_cann_profile_project",
    "CANN_BENCH_SRC_DIR",
]
