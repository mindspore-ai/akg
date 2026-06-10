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

from pathlib import Path
import sys

SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "workspace_autoresearch" / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from utils.baseline_anchor import current_fingerprint, fingerprint_mismatch  # noqa: E402
from phase_machine import Progress  # noqa: E402
from task_config import EvalOutcome, EvalResult  # noqa: E402
from workflow.progress_reducer import reduce_round_progress  # noqa: E402


def test_shape_aware_stored_baseline_mismatches_when_descs_disappear():
    stored = {"num_cases": 2, "shape_signature": "oldshape"}
    current = current_fingerprint(2)

    assert current == {"num_cases": 2, "shape_signature": None}
    assert fingerprint_mismatch(stored, current) == {
        "shape_signature": ("oldshape", None),
    }


def test_legacy_num_cases_only_fingerprint_still_matches_without_descs():
    stored = {"num_cases": 2}
    current = current_fingerprint(2)

    assert fingerprint_mismatch(stored, current) is None


def test_round_progress_refreshes_shape_metadata():
    progress = Progress(
        task="shape_task",
        baseline_metric=10.0,
        baseline_source="ref",
        baseline_fingerprint=current_fingerprint(2, ["old0", "old1"]),
        num_cases=2,
        per_shape_descs=["old0", "old1"],
    )
    metrics = {
        "num_cases": 2,
        "per_shape_descs": ["new0", "new1"],
        "ref_latency_us": 12.0,
        "per_shape_base_us": [11.0, 13.0],
    }
    result = EvalResult(outcome=EvalOutcome.OK, metrics=metrics)

    reduced = reduce_round_progress(
        progress, result, round_num=3, consecutive_failures=0,
        best_metric=5.0, best_commit="abc123",
    )

    assert reduced.progress.num_cases == 2
    assert reduced.progress.per_shape_descs == ["new0", "new1"]
    assert reduced.progress.baseline_fingerprint == current_fingerprint(
        2, ["new0", "new1"])
