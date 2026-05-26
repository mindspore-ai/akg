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

"""Shared config loader for framework reference tables.

Every backend/DSL/arch mapping, hallucinated-script alias, and worker-only
module list lives in `.autoresearch/config.yaml`. This module reads that
file once per process and exposes small typed accessors — callers never
hand-build these tables inside Python modules.
"""

# pylint: disable=missing-function-docstring
from functools import lru_cache
import os
import re
from typing import Dict

import yaml

# __file__ now lives in scripts/utils/; climb two levels to reach .autoresearch/.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AUTORESEARCH_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
_CONFIG_PATH = os.path.join(_AUTORESEARCH_DIR, "config.yaml")
_CODE_CHECKER_PATH = os.path.join(_AUTORESEARCH_DIR, "code_checker.yaml")


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    return data


@lru_cache(maxsize=1)
def _raw() -> dict:
    """Load config.yaml once. Missing file is a hard error — the framework
    ships with one and several modules depend on it."""
    return _load_yaml(_CONFIG_PATH)


@lru_cache(maxsize=1)
def _code_checker_raw() -> dict:
    """Load code_checker.yaml once. Missing file / key is a hard error:
    the checker has no fallback defaults, by design."""
    return _load_yaml(_CODE_CHECKER_PATH)


def default_dsl() -> str:
    return str(_raw().get("default_dsl", "triton_ascend"))


def worker_only_modules() -> frozenset:
    return frozenset(_raw().get("worker_only_modules", []))


def hallucinated_scripts() -> Dict[str, str]:
    return dict(_raw().get("hallucinated_scripts", {}))


# Thin code_checker.yaml accessors. Missing keys → KeyError naturally.
# code_checker.py caches these into module-level constants at its import;
# callers should not invoke these repeatedly.

def code_checker_hard_ops() -> frozenset:
    return frozenset(_code_checker_raw()["hard_ops"])


def code_checker_soft_ops() -> frozenset:
    return frozenset(_code_checker_raw()["soft_ops"])


def code_checker_triton_decorators() -> frozenset:
    return frozenset(_code_checker_raw()["triton_decorators"])


def code_checker_torch_call_prefixes() -> frozenset:
    return frozenset(_code_checker_raw()["torch_call_prefixes"])


def code_checker_kernel_class_name() -> str:
    return _code_checker_raw()["kernel_class_name"]


def code_checker_kernel_forward_method() -> str:
    return _code_checker_raw()["kernel_forward_method"]


def code_checker_triton_module_name() -> str:
    return _code_checker_raw()["triton_module_name"]


def code_checker_dsl_compliance_prefix() -> str:
    return _code_checker_raw()["dsl_compliance_prefix"]


def code_checker_stray_text_re() -> "re.Pattern[str]":
    """Regex matching a run of `min_run` consecutive chars in any unicode_range."""
    cfg = _code_checker_raw()["stray_text"]
    cls = "".join(f"\\u{lo:04x}-\\u{hi:04x}" for lo, hi in cfg["unicode_ranges"])
    return re.compile(f"[{cls}]{{{cfg['min_run']},}}")


def code_checker_autotune_re() -> "re.Pattern[str]":
    """Regex matching `@<triton_module>.<decorator_attr>(`."""
    triton_mod = _code_checker_raw()["triton_module_name"]
    deco = _code_checker_raw()["autotune"]["decorator_attr"]
    return re.compile(rf"@{re.escape(triton_mod)}\.{re.escape(deco)}\s*\(", re.MULTILINE)


def code_checker_restore_value_re() -> "re.Pattern[str]":
    """Regex matching the required autotune kwarg assignment."""
    kwarg = _code_checker_raw()["autotune"]["required_kwarg"]
    return re.compile(rf"{re.escape(kwarg)}\s*=")
