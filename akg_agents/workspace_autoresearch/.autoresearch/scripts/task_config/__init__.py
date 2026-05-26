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

"""task_config — TaskConfig dataclass + metric/outcome helpers.

The eval execution path now lives in `utils.akg_eval` (sync bridge into
``akg_agents.op.verifier.KernelVerifier`` + WorkerManager); this package
only owns parsing and metric semantics.
"""
from .loader import TaskConfig, load_task_config
from .metric_policy import (
    EvalOutcome, EvalResult, check_constraints, is_improvement,
    format_result_summary,
)
