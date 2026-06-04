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

"""Single source for paths that live OUTSIDE this scripts/ package.

WA reuses akg_agents' skill tree directly (no vendored copy):

    workspace_autoresearch/scripts/         <- this package
    ../python/akg_agents/op/resources/skills/  <- per-DSL skill markdown

`AKG_AGENTS_AR_SKILLS_ROOT` (set in .claude/settings.json) overrides the
relative fallback so the resolution works both when the slash command
runs from this dir and when callers cd elsewhere.

CA's `eval_dir()` (its vendored eval package) has no analogue here: the
verifier lives at `akg_agents.op.verifier` and is reached via
``utils.akg_eval``, not via a filesystem path.
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_WS_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
# workspace_autoresearch -> akg_agents -> python/akg_agents/op/resources/skills
_DEFAULT_SKILLS = os.path.abspath(os.path.join(
    _WS_ROOT, "..", "python", "akg_agents", "op", "resources", "skills"))


def skills_dir() -> str:
    """Per-DSL skill documentation tree. Reads AKG_AGENTS_AR_SKILLS_ROOT if
    set; falls back to the relative path under akg_agents."""
    return os.environ.get("AKG_AGENTS_AR_SKILLS_ROOT", _DEFAULT_SKILLS)


def latency_refs_dir() -> str:
    """Back-compat alias for the skills tree root. guidance.py expands the
    per-phase prompt Globs against this path."""
    return skills_dir()
