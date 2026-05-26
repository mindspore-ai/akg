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

"""workflow/ — orchestration layer between hooks and state_store.

Owns the rules that turn "what just happened" into "next phase" and the
record_round / run_baseline_init bodies. Both are called in-process by
engine/pipeline.py and engine/baseline.py respectively; the previous
shell wrappers (keep_or_discard.py, _baseline_init.py) have been
deleted now that no caller crosses a subprocess boundary.
"""
from .transition import PhaseController
from .planning import PlanStore
from .baseline import run_baseline_init
from .round import record_round

__all__ = ["PhaseController", "PlanStore", "run_baseline_init", "record_round"]
