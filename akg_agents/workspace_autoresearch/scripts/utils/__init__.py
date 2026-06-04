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

"""utils/ — stateless library modules imported by engine/, hooks/,
phase_machine/, workflow/, task_config/, and batch/.

No CLI entry points live here. The Triton regression check (no
@triton.jit / forbidden torch.* in forward / etc.) is delivered by
`akg_agents.op.utils.code_checker.CodeChecker` directly — quick_check
and batch/verify call it without a WA-side wrapper.

Nothing in this package mutates state. Splitting them out makes the
dependency direction obvious: utils sits at the bottom of the stack and
never imports from any sibling package.

---------------------------------------------------------------------------
Invariant — IMPORT STYLE INSIDE utils/

When a module in utils/ imports another utils/ module, use the
relative form: `from .settings import …`, NOT the absolute
`from settings import …`. The absolute form silently relies on
`scripts/utils/` being in sys.path — daemons (worker, batch driver)
add only `scripts/`, so the absolute form works in ad-hoc CLI runs
but blows up the first time a long-running process imports the
module.

If you're adding a new utils module that needs another utils module:
always use `from .<sibling> import X`. Audit your callsite for the
absolute form before pushing.
---------------------------------------------------------------------------
"""
