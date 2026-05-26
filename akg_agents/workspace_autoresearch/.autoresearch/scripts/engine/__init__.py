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

"""engine/ — orchestration scripts the LLM and hooks invoke via subprocess.

Holds the blessed-script set: pipeline.py (main post-edit driver) plus the
single-purpose CLIs it spawns (quick_check, eval_wrapper, settle), the
BASELINE-phase entry (baseline.py), the PLAN-phase entry (create_plan.py),
and the /autoresearch arg dispatcher (parse_args.py).

Body-level logic (record_round, run_baseline_init) lives in workflow/
and is now called in-process; the earlier shell wrappers
(keep_or_discard.py, _baseline_init.py) have been deleted because every
caller went through workflow.* directly.

These are CLIs, not a library — they are exec'd via Bash, not imported.
The package marker exists so cross-package imports (e.g. phase_machine
referencing engine.quick_check.check_editable_files) resolve cleanly.
"""
