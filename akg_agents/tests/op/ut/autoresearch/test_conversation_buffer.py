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

"""
Tests for ConversationBuffer — the single owner of the autoresearch
message list (P2 refactor).

Before P2 the message list was directly mutated by AgentLoop, TurnExecutor,
compress.py, and llm_client.py. ConversationBuffer centralizes ownership
and exposes explicit semantics for read/append/replace/microcompact/
auto_compact/save/load. These tests cover the public API plus the
critical edge cases (view-is-snapshot, append_to_first no-op safety,
auto_compact no-op signal).
"""

import json
import os

import pytest

from akg_agents.op.autoresearch.agent.conversation import ConversationBuffer


# ---------------------------------------------------------------------------
# Basic read / write / view semantics
# ---------------------------------------------------------------------------


class TestBasicOperations:
    def test_starts_empty(self):
        buf = ConversationBuffer()
        assert len(buf) == 0
        assert buf.is_empty()
        assert buf.view() == []

    def test_append_increases_len(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "hi"})
        assert len(buf) == 1
        assert not buf.is_empty()
        assert buf.view() == [{"role": "user", "content": "hi"}]

    def test_extend_appends_batch(self):
        buf = ConversationBuffer()
        buf.extend([
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ])
        assert len(buf) == 2
        assert buf.view()[0]["content"] == "a"
        assert buf.view()[1]["content"] == "b"

    def test_replace_atomic(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "old"})
        buf.replace([{"role": "user", "content": "new"}])
        assert len(buf) == 1
        assert buf.view() == [{"role": "user", "content": "new"}]

    def test_replace_defensive_copy(self):
        """replace() must not retain a reference to the input list — the
        caller can keep mutating their list without affecting the buffer."""
        buf = ConversationBuffer()
        external = [{"role": "user", "content": "x"}]
        buf.replace(external)
        external.append({"role": "user", "content": "y"})  # mutate caller's list
        assert len(buf) == 1
        assert buf.view() == [{"role": "user", "content": "x"}]

    def test_clear(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "hi"})
        buf.clear()
        assert buf.is_empty()
        assert buf.view() == []


# ---------------------------------------------------------------------------
# view() returns a snapshot — protects against accidental mutation
# ---------------------------------------------------------------------------


class TestViewSnapshot:
    def test_view_is_a_copy(self):
        """Mutating view()'s result must NOT affect the buffer."""
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "a"})
        snapshot = buf.view()
        snapshot.append({"role": "user", "content": "b"})  # caller mutates
        assert len(buf) == 1
        assert buf.view() == [{"role": "user", "content": "a"}]

    def test_view_per_call_returns_fresh_list(self):
        """Each call returns its own list — they're independent."""
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "a"})
        v1 = buf.view()
        v2 = buf.view()
        assert v1 == v2
        assert v1 is not v2


# ---------------------------------------------------------------------------
# tail_since() — for incremental save
# ---------------------------------------------------------------------------


class TestTailSince:
    def test_tail_since_zero_returns_all(self):
        buf = ConversationBuffer()
        buf.extend([
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ])
        assert buf.tail_since(0) == buf.view()

    def test_tail_since_partial(self):
        buf = ConversationBuffer()
        buf.extend([
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ])
        assert buf.tail_since(1) == [
            {"role": "user", "content": "b"},
            {"role": "user", "content": "c"},
        ]

    def test_tail_since_at_end(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "a"})
        assert buf.tail_since(1) == []

    def test_tail_since_past_end(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "a"})
        assert buf.tail_since(99) == []


# ---------------------------------------------------------------------------
# append_to_first — used by resume_info graft
# ---------------------------------------------------------------------------


class TestAppendToFirst:
    def test_append_to_first_appends_text(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "initial"})
        buf.append_to_first(" appended")
        assert buf.view()[0]["content"] == "initial appended"

    def test_append_to_first_noop_on_empty_buffer(self):
        buf = ConversationBuffer()
        buf.append_to_first("ignored")  # must not raise
        assert buf.is_empty()

    def test_append_to_first_noop_when_first_content_is_not_str(self):
        """The first message could legitimately have a list content (tool
        results); append_to_first should silently no-op rather than crash."""
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": [{"type": "tool_result", "content": "x"}]})
        buf.append_to_first(" extra")  # must not raise
        # Content is unchanged
        assert isinstance(buf.view()[0]["content"], list)


# ---------------------------------------------------------------------------
# microcompact — delegates to compress.microcompact, mutates in place
# ---------------------------------------------------------------------------


class TestMicrocompact:
    def test_microcompact_clears_old_tool_results(self):
        """microcompact should replace old tool_result content with [cleared]."""
        long_content = "x" * 500
        buf = ConversationBuffer()
        buf.extend([
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "1", "content": long_content},
            ]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "2", "content": long_content},
            ]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "3", "content": long_content},
            ]},
        ])
        buf.microcompact(min_chars=100, keep_recent=1)
        view = buf.view()
        # Last tool result is preserved (keep_recent=1)
        assert view[-1]["content"][0]["content"] == long_content
        # Older tool results are cleared
        assert view[0]["content"][0]["content"] == "[cleared]"
        assert view[2]["content"][0]["content"] == "[cleared]"

    def test_microcompact_noop_when_few_tool_results(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "1", "content": "x" * 500},
        ]})
        buf.microcompact(min_chars=100, keep_recent=1)
        # Single tool result preserved (keep_recent=1)
        assert buf.view()[0]["content"][0]["content"] == "x" * 500


# ---------------------------------------------------------------------------
# Persistence — save/load roundtrip
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_latest_then_load(self, tmp_path):
        task_dir = str(tmp_path)
        buf = ConversationBuffer()
        buf.extend([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "third"},
        ])
        buf.save_latest(task_dir, session_dir="agent_session")

        buf2 = ConversationBuffer()
        loaded = buf2.load_latest(task_dir, session_dir="agent_session")
        assert loaded is True
        assert len(buf2) == 3
        assert buf2.view()[0]["content"] == "first"
        assert buf2.view()[2]["content"] == "third"

    def test_load_latest_returns_false_when_missing(self, tmp_path):
        buf = ConversationBuffer()
        loaded = buf.load_latest(str(tmp_path), session_dir="agent_session")
        assert loaded is False
        assert buf.is_empty()

    def test_save_full_increment_advances_cursor(self, tmp_path):
        task_dir = str(tmp_path)
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "a"})
        buf.append({"role": "user", "content": "b"})

        # First save: cursor=0, should write 2 messages
        new_cursor = buf.save_full_increment(
            task_dir, session_dir="agent_session", since_idx=0)
        assert new_cursor == 2

        # Add more
        buf.append({"role": "user", "content": "c"})

        # Second save: cursor=2, should write only the new message
        new_cursor = buf.save_full_increment(
            task_dir, session_dir="agent_session", since_idx=new_cursor)
        assert new_cursor == 3

        # Verify file contains all 3 messages (append mode)
        path = os.path.join(task_dir, "agent_session", "messages",
                            "messages_full.jsonl")
        with open(path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == 3
        assert lines[0]["content"] == "a"
        assert lines[2]["content"] == "c"

    def test_save_full_increment_skips_when_no_new_msgs(self, tmp_path):
        task_dir = str(tmp_path)
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "a"})
        new_cursor = buf.save_full_increment(
            task_dir, session_dir="agent_session", since_idx=1)
        # since_idx already past end → nothing to write
        assert new_cursor == 1
        path = os.path.join(task_dir, "agent_session", "messages",
                            "messages_full.jsonl")
        assert not os.path.exists(path)


# ---------------------------------------------------------------------------
# Skill injection lifecycle
# ---------------------------------------------------------------------------


class TestSkillReadLifecycle:
    """Tests for track_item_skill_read / unload_item_reads / on_buffer_rebuilt.

    Skills are no longer auto-injected; the agent read_file's SKILL.md
    on demand, and the buffer elides the read's tool_result body when
    the owning plan item settles.
    """

    @staticmethod
    def _tool_result_msg(tool_use_id: str, text: str) -> dict:
        """Real shape produced by TurnExecutor._dispatch_tools —
        tool_result blocks are nested inside a user message's content
        list, never top-level."""
        return {
            "role": "user",
            "content": [{
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": text,
            }],
        }

    def _buf_with_reads(self):
        buf = ConversationBuffer()
        buf.append({"role": "user", "content": "msg-0"})
        buf.append(self._tool_result_msg("tu-1", "skill-A body"))
        buf.append(self._tool_result_msg("tu-2", "skill-B body"))
        buf.append(self._tool_result_msg("tu-other", "unrelated read"))
        buf.track_item_skill_read("tu-1", "p1")
        buf.track_item_skill_read("tu-2", "p2")
        return buf

    def test_unload_elides_only_tracked_item_reads(self):
        buf = self._buf_with_reads()
        n = buf.unload_item_reads("p1")
        assert n == 1
        msgs = buf.view()
        # tu-1's block was flipped; the others untouched
        assert msgs[1]["content"][0]["content"] == buf._SKILL_READ_ELIDED
        assert msgs[2]["content"][0]["content"] == "skill-B body"
        assert msgs[3]["content"][0]["content"] == "unrelated read"

    def test_unload_is_idempotent_and_clears_tracking(self):
        buf = self._buf_with_reads()
        buf.unload_item_reads("p1")
        assert buf.unload_item_reads("p1") == 0

    def test_unload_unknown_item_is_noop(self):
        buf = self._buf_with_reads()
        assert buf.unload_item_reads("missing") == 0
        assert buf.unload_item_reads("p1") == 1

    def test_track_ignores_empty_ids(self):
        buf = ConversationBuffer()
        buf.track_item_skill_read("", "p1")
        buf.track_item_skill_read("tu-1", "")
        assert buf.unload_item_reads("p1") == 0

    def test_on_buffer_rebuilt_keeps_tracking_for_surviving_tool_results(self):
        """Phase 5 [Medium]: auto_compact carries recent rounds
        forward, so tool_results still in _msgs must stay unloadable.
        ``on_buffer_rebuilt`` intersects _item_read_ids with the
        tool_use_ids still present in _msgs; nothing was dropped
        here, so p1/p2 reads remain elide-able."""
        buf = self._buf_with_reads()
        buf.on_buffer_rebuilt()
        assert buf.unload_item_reads("p1") == 1
        assert buf.unload_item_reads("p2") == 1

    def test_on_buffer_rebuilt_drops_tracking_for_compacted_tool_results(self):
        """If the rebuild actually dropped the tool_results (say a
        summary replaces them), the tracking entries are purged —
        there's no id left to match."""
        buf = self._buf_with_reads()
        # Replace with messages that carry no tool_result at all.
        buf.replace([
            {"role": "user", "content": "[SUMMARY] all reads compacted"},
        ])
        buf.on_buffer_rebuilt()
        assert buf.unload_item_reads("p1") == 0
        assert buf.unload_item_reads("p2") == 0
