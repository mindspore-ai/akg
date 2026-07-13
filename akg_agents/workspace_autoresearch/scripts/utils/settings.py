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

"""Shared accessors for config.yaml.

config.yaml is the SINGLE SOURCE OF TRUTH for the framework reference
tables and tunable knobs below. This module reads it once per process and
exposes typed accessors. There are NO in-code defaults: a missing section
or key is a hard error, because config.yaml ships with every key present.
Retune by editing config.yaml — never by editing values here.
"""
from functools import lru_cache
import os
from typing import Dict

import yaml

# __file__ now lives in scripts/utils/; climb two levels to reach autoresearch/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTORESEARCH_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CONFIG_PATH = os.path.join(_AUTORESEARCH_DIR, "config.yaml")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    return data


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Load config.yaml once. Missing file is a hard error — the framework
    ships with one and every accessor below depends on it."""
    return _load_yaml(_CONFIG_PATH)


def _get(section: str, key: str):
    """Read config.yaml[section][key]. config.yaml is the single source of
    truth, so a missing section/key raises — there is no in-code default."""
    sect = _raw().get(section)
    if not isinstance(sect, dict) or key not in sect:
        raise KeyError(
            f"{_CONFIG_PATH}: missing required key '{section}.{key}'")
    return sect[key]


def hallucinated_scripts() -> Dict[str, str]:
    return dict(_raw().get("hallucinated_scripts", {}))


# --- task defaults -----------------------------------------------------
def default_max_rounds() -> int:
    """Default optimization-round budget when a task doesn't specify one.
    Single source for scaffold (new task.yaml) and loader (TaskConfig
    fallback) so the two cannot drift."""
    return _get("defaults", "max_rounds")


def default_eval_timeout() -> int:
    """Per-shape verify/profile budget (seconds) when a task omits it."""
    return _get("defaults", "eval_timeout")


def default_reference_data_timeout() -> int:
    """Reference-data generation budget (seconds) when omitted."""
    return _get("defaults", "reference_data_timeout")


def default_smoke_test_timeout() -> int:
    """quick_check smoke-test budget (seconds) when a task omits it."""
    return _get("defaults", "smoke_test_timeout")


def default_code_checker_enabled() -> bool:
    """Whether the triton-impl AST regression check runs by default."""
    return bool(_get("defaults", "code_checker_enabled"))


def default_metric() -> dict:
    """Primary-metric defaults (primary / lower_is_better /
    improvement_threshold). scaffold writes these into a new task.yaml;
    loader falls back to them when the task.yaml omits the metric block."""
    m = _get("defaults", "metric")
    if not isinstance(m, dict):
        raise ValueError(f"{_CONFIG_PATH}: 'defaults.metric' must be a mapping")
    return m


def skill_dsl() -> str:
    """Kebab-case name of the skills/ subtree this repo consults
    (`triton-ascend` / `triton-cuda` / `pypto` / `cpp` / `cuda-c` /
    `tilelang-cuda` / `tilelang-ascend` /
    `ascendc` / `ascendc-catlass`). Used by guidance.py
    to expand `<dsl>` in PLAN /
    REPLAN / DIAGNOSE prompt Glob patterns to a literal directory name.
    Single-DSL-per-repo by design; lives at defaults/skill_dsl in
    config.yaml, not on per-task TaskConfig."""
    return str(_get("defaults", "skill_dsl"))


# --- workspace target triple ------------------------------------------
# Pinned per repo; same single-target-per-repo design as skill_dsl. The
# bridge in utils.akg_eval passes these straight to KernelVerifier; the
# upstream CA equivalent is eval_kernel.py's argparse defaults.
def target_backend() -> str:
    return str(_get("defaults", "backend"))


def target_framework() -> str:
    return str(_get("defaults", "framework"))


def target_dsl() -> str:
    """Snake-form DSL (`triton_ascend`) — what KernelVerifier consumes.
    Distinct from `skill_dsl()` (kebab) which names a directory."""
    return str(_get("defaults", "dsl"))


# --- eval timing measurement (read where the timing runs: on remote eval
#     that is the WORKER's config.yaml) ----------------------------------
def eval_warmup() -> int:
    return _get("eval", "warmup")


def eval_repeats() -> int:
    return _get("eval", "repeats")


# Worker port / readiness timing 之前在这里有 worker_port / worker_ready_*
# 访问器，现已下线 —— akg_cli 全程从 ``cli/service/worker_config.WorkerConfig``
# 一处读 yaml，workspace 这边没有直接调用 worker.* 字段的脚本。如果需要
# 在 WA 脚本读 worker.*，请改用 ``from akg_agents.cli.service.worker_config
# import WorkerConfig; cfg = WorkerConfig.load()`` 走单一事实源。


# --- batch pre-flight verification timeouts (seconds) ------------------
def batch_tier1_timeout() -> int:
    return _get("batch", "tier1_timeout")


def batch_tier2_timeout() -> int:
    return _get("batch", "tier2_timeout")


# --- batch driver knobs (overridable via run.py CLI flags) -------------
def batch_run_timeout_min() -> int:
    """Hard wall-clock cap per op in minutes (batch/run.py --timeout-min)."""
    return _get("batch", "run_timeout_min")


def batch_cooldown_sec() -> int:
    """Seconds to sleep between ops (batch/run.py --cooldown-sec)."""
    return _get("batch", "cooldown_sec")


def batch_transient_retries() -> int:
    """Max times batch/run.py re-spawns `claude --print /autoresearch
    --resume` for the same op after a transient claude.exe crash
    (rc != 0 with framework progress intact). 0 disables retry."""
    return _get("batch", "transient_retries")


# --- phase machine thresholds -----------------------------------------
def consecutive_fail_threshold() -> int:
    """FAIL-streak length that flips PLAN/EDIT into DIAGNOSE."""
    return _get("defaults", "consecutive_fail_threshold")


def diagnose_max_attempts() -> int:
    """DIAGNOSE subagent retries allowed per plan_version before the
    manual-fallback artifact path opens."""
    return _get("defaults", "diagnose_max_attempts")


# --- resume heartbeat freshness window (seconds) ----------------------
def heartbeat_fresh_seconds() -> int:
    return _get("resume", "heartbeat_fresh_seconds")


# --- speedup classification thresholds (x vs ref) ---------------------
def speedup_improved_above() -> float:
    return _get("metrics", "speedup_improved_above")


def speedup_regress_below() -> float:
    return _get("metrics", "speedup_regress_below")


def classify_speedup(v: float) -> str:
    """'improved' / 'on-par' / 'regress' per the configured thresholds.
    Single owner for the batch reporters (summarize, monitor)."""
    if v > speedup_improved_above():
        return "improved"
    if v < speedup_regress_below():
        return "regress"
    return "on-par"


def recorded_speedup(src) -> float | None:
    """THE single reader for the recorded speedup — ``best_speedup``, the
    per-shape-ratio geomean produced once by ``aggregate.geomean_ratio`` and
    stored in task state. Accepts a state/result dict or a progress object.
    Returns None when unset / non-positive. Consumers read speedup from here;
    they must NOT re-derive it from baseline/best latencies (a different, wrong
    definition — that ratio is mean(base)/mean(gen), not the geomean of ratios)."""
    v = src.get("best_speedup") if hasattr(src, "get") \
        else getattr(src, "best_speedup", None)
    return float(v) if isinstance(v, (int, float)) and v > 0 else None
