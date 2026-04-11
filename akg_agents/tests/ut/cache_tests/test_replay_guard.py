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

"""UT: ReplaySnapshotStore record / replay / guard logic."""

import json
import tempfile

import pytest

from akg_agents.core_v2.langgraph_base.replay_guard import (
    ReplayGuardError,
    ReplaySnapshotStore,
    _fingerprint,
    _snap_value,
    _match,
    _is_empty,
    GUARD_FIELDS,
    _NODE_SNAPSHOTS_KEY,
)


@pytest.fixture
def cache_file():
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        f.write("{}")
        f.flush()
        yield f.name


class TestSnapValueAndMatch:
    def test_short_string_kept_verbatim(self):
        assert _snap_value("verifier_result", True) is True
        assert _snap_value("conductor_decision", "coder") == "coder"

    def test_long_string_hashed(self):
        code = "x" * 300
        snap = _snap_value("coder_code", code)
        assert isinstance(snap, dict)
        assert "__hash" in snap
        assert snap["__len"] == 300

    def test_match_hash(self):
        code = "x" * 300
        snap = {"__hash": _fingerprint(code), "__len": 300}
        assert _match("coder_code", snap, code)
        assert not _match("coder_code", snap, "y" * 300)

    def test_match_bool(self):
        assert _match("verifier_result", True, True)
        assert not _match("verifier_result", True, False)

    def test_match_none(self):
        assert _match("some_field", None, None)
        assert not _match("some_field", None, "something")

    def test_match_none_vs_empty_string(self):
        assert _match("codegen_invalid_reason", None, "")
        assert _match("codegen_invalid_reason", "", None)
        assert _match("some_field", None, [])
        assert _match("some_field", [], None)


class TestReplaySnapshotStore:
    def test_record_and_replay_pass(self, cache_file):
        store = ReplaySnapshotStore(cache_file, mode="record")
        result = {
            "coder_code": "class ModelNew: pass",
            "codegen_invalid": False,
            "codegen_invalid_reason": None,
        }
        store.record("coder", step=1, result=result)

        replay_store = ReplaySnapshotStore(cache_file, mode="replay")
        assert replay_store.has_snapshots
        replay_store.verify("coder", step=1, result=result)

    def test_record_and_replay_mismatch_raises(self, cache_file):
        store = ReplaySnapshotStore(cache_file, mode="record")
        result = {
            "verifier_result": True,
            "verifier_error": "",
        }
        store.record("verifier", step=2, result=result)

        replay_store = ReplaySnapshotStore(cache_file, mode="replay")
        bad_result = {
            "verifier_result": False,
            "verifier_error": "env error: compiler not found",
        }
        with pytest.raises(ReplayGuardError) as exc_info:
            replay_store.verify("verifier", step=2, result=bad_result)
        assert "verifier_result" in str(exc_info.value)
        assert exc_info.value.field == "verifier_result"

    def test_unknown_node_ignored(self, cache_file):
        store = ReplaySnapshotStore(cache_file, mode="record")
        store.record("unknown_node", step=0, result={"foo": "bar"})
        assert not store.has_snapshots

    def test_missing_snapshot_warns_not_raises(self, cache_file):
        store = ReplaySnapshotStore(cache_file, mode="replay")
        store.verify("coder", step=99, result={"coder_code": "x"})

    def test_conductor_decision_guard(self, cache_file):
        store = ReplaySnapshotStore(cache_file, mode="record")
        result = {
            "conductor_decision": "coder",
            "conductor_suggestion": "fix the import",
        }
        store.record("conductor", step=3, result=result)

        replay_store = ReplaySnapshotStore(cache_file, mode="replay")
        replay_store.verify("conductor", step=3, result=result)

        bad = {
            "conductor_decision": "finish",
            "conductor_suggestion": "fix the import",
        }
        with pytest.raises(ReplayGuardError) as exc_info:
            replay_store.verify("conductor", step=3, result=bad)
        assert exc_info.value.field == "conductor_decision"

    def test_long_code_hash_comparison(self, cache_file):
        long_code = "import torch\n" * 100 + "class ModelNew:\n    pass\n"
        store = ReplaySnapshotStore(cache_file, mode="record")
        store.record("coder", step=0, result={
            "coder_code": long_code,
            "codegen_invalid": False,
            "codegen_invalid_reason": None,
        })

        replay_store = ReplaySnapshotStore(cache_file, mode="replay")
        replay_store.verify("coder", step=0, result={
            "coder_code": long_code,
            "codegen_invalid": False,
            "codegen_invalid_reason": None,
        })

        with pytest.raises(ReplayGuardError):
            replay_store.verify("coder", step=0, result={
                "coder_code": long_code + "# modified",
                "codegen_invalid": False,
                "codegen_invalid_reason": None,
            })

    def test_snapshots_persisted_in_cache_file(self, cache_file):
        store = ReplaySnapshotStore(cache_file, mode="record")
        store.record("verifier", step=0, result={
            "verifier_result": True,
            "verifier_error": "",
        })
        with open(cache_file, "r") as f:
            data = json.load(f)
        assert _NODE_SNAPSHOTS_KEY in data
        assert "verifier@0" in data[_NODE_SNAPSHOTS_KEY]

    def test_invalid_mode_raises(self, cache_file):
        with pytest.raises(ValueError, match="record/replay"):
            ReplaySnapshotStore(cache_file, mode="off")
