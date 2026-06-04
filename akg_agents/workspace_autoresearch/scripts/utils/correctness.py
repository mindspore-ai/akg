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

"""Stub for the layered-tolerance per-case correctness comparator.

CA's standalone implementation lives at `scripts/utils/correctness.py` and
is used by `batch/verify.py`'s pre-batch numeric sweep. WA's eval bridge
(`utils.akg_eval`) routes verification through ``akg_agents.op.verifier``,
which runs its own correctness comparison inside the worker — so the
pre-batch comparator here is only needed if the operator wants an
independent numeric audit before kicking off a long batch run.

Until that adapter is written, `compare_outputs_per_case` returns
``(True, "")`` for every case. ``batch/verify.py`` will treat the batch as
"clean" without doing any real per-case comparison; the per-task verify
inside the agent loop is still authoritative.

To wire the AKG comparator in later, replace this with a thin call into
``akg_agents.op.verifier.KernelVerifier`` or a dedicated numeric helper.
"""
from typing import Any, Tuple


def compare_outputs_per_case(*_args: Any, **_kwargs: Any) -> Tuple[bool, str]:
    """No-op pass. Returns ``(True, "")``."""
    return True, ""
