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

"""Runtime-guard policy, loaded from the shared CodeChecker config.

Single source of truth is ``akg_agents/op/config/code_checker.yaml`` under
``ascendc_anti_cheat.runtime_guard`` — the same file the static CodeChecker
reads, so the static and runtime anti-cheat blacklists never drift. Missing or
malformed keys surface as KeyError / TypeError on first access (no fallback
defaults), matching ``op.utils.code_checker``.
"""

import importlib.resources

import yaml

with importlib.resources.files("akg_agents.op.config").joinpath(
    "code_checker.yaml"
).open("r", encoding="utf-8") as _f:
    _RG = yaml.safe_load(_f)["ascendc_anti_cheat"]["runtime_guard"]

# Enforcement mode default; AKG_GUARD_MODE env var overrides at runtime.
DEFAULT_MODE: str = _RG["default_mode"]

# ATen core-compute leaves the runtime gate disables during the candidate forward.
COMPUTE_LEAVES = tuple(_RG["compute_leaves"])
