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

"""
mfusion - MLIR Optimization Tools

This package provides MLIR-based optimization tools and dialect support.
"""

from pathlib import Path

from mfusion._mlir_libs._mfusion import *

__version__ = "1.0"


def _read_commit_id() -> str:
    commit_id_file = Path(__file__).with_name(".commit_id")
    try:
        return commit_id_file.read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"


__commit_id__ = _read_commit_id()


def get_build_info():
    """Return installed package build metadata."""
    return {"version": __version__, "commit_id": __commit_id__}
