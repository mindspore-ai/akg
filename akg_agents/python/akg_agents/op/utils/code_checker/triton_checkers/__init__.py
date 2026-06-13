# Copyright 2025 Huawei Technologies Co., Ltd
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

"""Non-blocking Triton diagnostic checks used by CodeChecker."""

from akg_agents.op.utils.code_checker.triton_checkers.api_signature_checker import (
    CHECKER_NAME as API_SIGNATURE_CHECKER,
    TritonApiSignatureChecker,
)
from akg_agents.op.utils.code_checker.triton_checkers.high_confidence_semantics_checker import (
    CHECKER_NAME as HIGH_CONFIDENCE_SEMANTICS_CHECKER,
    TritonHighConfidenceSemanticsChecker,
)
from akg_agents.op.utils.code_checker.registry import CheckerSpec
from akg_agents.op.utils.code_checker.triton_checkers.runner import (
    TritonCheckerRunner,
    format_issues_text,
    issues_to_dicts,
    run_diagnostic_checkers,
)

__all__ = [
    "API_SIGNATURE_CHECKER",
    "HIGH_CONFIDENCE_SEMANTICS_CHECKER",
    "TritonApiSignatureChecker",
    "TritonCheckerRunner",
    "TritonHighConfidenceSemanticsChecker",
    "default_triton_checkers",
    "format_issues_text",
    "issues_to_dicts",
    "register_triton_checkers",
    "run_diagnostic_checkers",
]


def default_triton_checkers():
    """Default non-blocking Triton checker instances in execution order."""
    return [
        TritonApiSignatureChecker(),
        TritonHighConfidenceSemanticsChecker(),
    ]


def register_triton_checkers(registry, *, backend: str, dsl: str, config: dict | None) -> None:
    """Register non-blocking Triton diagnostics in their default order."""
    del backend, dsl, config
    registry.register(
        CheckerSpec(
            name=TritonApiSignatureChecker.name,
            group="triton",
            factory=TritonApiSignatureChecker,
        )
    )
    registry.register(
        CheckerSpec(
            name=TritonHighConfidenceSemanticsChecker.name,
            group="triton",
            factory=TritonHighConfidenceSemanticsChecker,
        )
    )
