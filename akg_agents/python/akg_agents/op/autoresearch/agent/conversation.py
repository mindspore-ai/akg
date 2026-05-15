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
ConversationBuffer — Single owner of the autoresearch message list.

Before P2 the conversation message list (`_messages`) had no clear
owner: AgentLoop initialized it but TurnExecutor mutated it directly,
compress.py replaced it with a new list, and the LLM adapter mutated
it in-place via ``append_assistant``. Four call sites all touched the
same list with different semantics (in-place vs replace), making it
hard to reason about message lifecycle and adding rollback safe-points.

ConversationBuffer fixes that by being the sole owner of the list.
All callers receive a buffer instance and use its public methods:

- read:      ``view()``, ``__len__()``, ``tail_since(idx)``
- write:     ``append(msg)``, ``replace(new_msgs)``, ``clear()``
- edit:      ``append_to_first(text)`` (resume_info graft)
- compress:  ``microcompact()``, ``auto_compact()``, ``force_rebuild()``
- persist:   ``save_full_increment()``, ``save_latest()``, ``load_latest()``

The compress module's pure list helpers stay unchanged — buffer
methods are thin wrappers that delegate and apply the result via
``replace()`` / direct mutation of the internal ``_msgs`` list.
"""

import re


class ConversationBuffer:
    """Single owner of the autoresearch conversation message list.

    All mutations go through this class. AgentLoop holds an instance;
    TurnExecutor / compress.py / the LLM adapter all RECEIVE the buffer
    instead of the raw list.

    Provides clear semantics for:
      - append (single message) / extend (batch)
      - view (read-only snapshot for LLM call / token estimate)
      - microcompact (in-place trim of old tool_results)
      - auto_compact (LLM summarization, replaces buffer atomically)
      - force_rebuild (emergency boundary, replaces buffer atomically)
      - save_full_increment / save_latest / load_latest (persistence)
    """

    def __init__(self) -> None:
        self._msgs: list[dict] = []

    # -- Read operations ---------------------------------------------------

    def view(self) -> list[dict]:
        """Return a shallow-copied snapshot of the message list.

        The copy protects against accidental shared-reference mutation
        during long-running LLM calls — the LLM call captures a stable
        view while the buffer can keep accepting new messages.

        Callers MUST NOT mutate the returned list to "reach into" the
        buffer; use the buffer's mutator methods instead.
        """
        return list(self._msgs)

    def __len__(self) -> int:
        return len(self._msgs)

    def is_empty(self) -> bool:
        return not self._msgs

    def tail_since(self, idx: int) -> list[dict]:
        """Return messages from index ``idx`` onwards.

        Used by the incremental save path: the loop tracks how many
        messages it has already flushed to ``messages_full.jsonl`` and
        only writes the new ones each turn.
        """
        return self._msgs[idx:]

    # -- Write operations --------------------------------------------------

    def append(self, msg: dict) -> None:
        self._msgs.append(msg)

    def extend(self, msgs: list[dict]) -> None:
        self._msgs.extend(msgs)

    def replace(self, new_msgs: list[dict]) -> None:
        """Atomically replace the entire buffer.

        Used by resume / compact / force_rebuild paths that need to
        substitute a fresh message list. Defensive copy so the caller
        can keep mutating its source list without affecting us.
        Automatically resets skill injection tracking.
        """
        self._msgs = list(new_msgs)
        self.on_buffer_rebuilt()

    def clear(self) -> None:
        """Empty the buffer and reset all skill-tracking maps.

        Routes through ``replace([])`` so ``on_buffer_rebuilt`` fires
        and the tracking invariant stays sealed inside this class —
        otherwise ``_skill_inject_keys`` / ``_item_inject_markers`` /
        ``_item_read_ids`` would survive a clear and cause dedup
        mis-hits or un-elideable residue on the next inject.
        """
        self.replace([])

    def append_to_first(self, text: str) -> None:
        """Append ``text`` to the content of the first message in the buffer.

        Used by the resume path to graft ``resume_info`` onto the
        bootstrap user message without rewriting the message list.
        Silently no-ops if the buffer is empty or the first message's
        content isn't a plain string.
        """
        if not self._msgs:
            return
        first = self._msgs[0]
        if isinstance(first.get("content"), str):
            first["content"] += text

    # -- Skill read lifecycle ----------------------------------------------
    #
    # Two parallel tracks feed skill content into the conversation:
    #
    #   (a) Voluntary read_file. Agent reads skills/<name>/SKILL.md
    #       on demand; the tool_result lives in the buffer like any
    #       other tool output. TurnExecutor calls
    #       ``track_item_skill_read`` to pin the tool_use_id to the
    #       active plan item so ``unload_item_reads`` can elide the
    #       body once the item settles.
    #
    #   (b) Auto-injection. When a plan item activates with a
    #       backing_skill, AgentLoop calls ``inject_backing_skill``
    #       before the next LLM call so the agent sees the SKILL.md
    #       body without having to read_file first. The inject is a
    #       plain user message prefixed with _SKILL_INJECT_PREFIX —
    #       ``unload_item_reads`` finds it by marker at settle time.
    #
    # Lifecycle methods:
    #   inject_backing_skill  — loop.py per-turn auto-inject (b).
    #   track_item_skill_read — turn.py registers (tool_use_id,
    #                           item_id) whenever the agent read_file's
    #                           a path under skills/ (a).
    #   unload_item_reads     — ``TurnExecutor.execute`` invokes this
    #                           with the just-settled item id (right
    #                           after ``feedback.settle_active`` closes
    #                           the item); elides both the (a)
    #                           tool_result bodies and the (b) inject
    #                           markers belonging to that item.
    #   on_buffer_rebuilt     — auto_compact / force_rebuild: rescans
    #                           the new ``_msgs`` to repopulate the
    #                           auto-inject dedup + marker map, and
    #                           filters ``_item_read_ids`` to only the
    #                           tool_use_ids that still appear as
    #                           tool_result blocks (reads the rebuild
    #                           threw away are purged from the map).

    # Elision markers and the synthetic-inject prefix. The inject
    # prefix is used as a needle at unload time so we can find the
    # right user message and swap its content without breaking the
    # Anthropic tool_use / tool_result pairing invariant.
    _SKILL_READ_ELIDED = "[skill read elided — plan item settled]"
    _SKILL_INJECT_PREFIX = "[skill auto-injected"

    def __init_read_tracking(self):
        if not hasattr(self, "_item_read_ids"):
            # item_id → set[tool_use_id]  (voluntary read_file)
            self._item_read_ids: dict[str, set[str]] = {}
        if not hasattr(self, "_item_inject_markers"):
            # item_id → set[marker_prefix]  (synthetic auto-injects)
            self._item_inject_markers: dict[str, set[str]] = {}

    def inject_backing_skill(
        self, item_id: str, skill_name: str, content: str,
        plan_version: int, max_chars: int = 6_000,
    ) -> bool:
        """Inject the backing_skill's SKILL.md content as a plain user
        message carrying a marker prefix.

        A tool_result would need a matching preceding tool_use in the
        same assistant message (Anthropic pairing rule); synthetic
        tool_use_ids don't satisfy that, so we use a user message
        instead. ``unload_item_reads`` evicts the content by matching
        the marker at settle time.
        """
        if not content:
            return False
        self.__init_read_tracking()
        key = (plan_version, item_id, skill_name)
        injected = getattr(self, "_skill_inject_keys", None)
        if injected is None:
            injected = set()
            self._skill_inject_keys = injected
        if key in injected:
            return False
        injected.add(key)
        sc = content
        if len(sc) > max_chars:
            sc = sc[:max_chars] + "\n...[truncated]"
        marker = (
            f"{self._SKILL_INJECT_PREFIX} for {item_id} v{plan_version} "
            f"({skill_name})]"
        )
        self.append({
            "role": "user",
            "content": f"{marker}\n{sc}",
        })
        self._item_inject_markers.setdefault(item_id, set()).add(marker)
        return True

    def track_item_skill_read(self, tool_use_id: str, item_id: str) -> None:
        """Register a read_file tool_use_id as belonging to ``item_id``.

        Called by TurnExecutor when the agent reads a ``skills/...``
        path while a plan item is active.
        """
        if not tool_use_id or not item_id:
            return
        self.__init_read_tracking()
        self._item_read_ids.setdefault(item_id, set()).add(tool_use_id)

    def unload_item_reads(self, item_id: str) -> int:
        """Elide every tracked skill content for ``item_id``.

        Two tracks:
          - synthetic injects (plain user messages starting with the
            inject marker) — content replaced wholesale.
          - voluntary read_file results (tool_result blocks nested
            inside a user message's content list) — block ``content``
            replaced, preserving the tool_use_id pairing.

        Returns the number of items edited (messages or blocks).
        """
        self.__init_read_tracking()
        edited = 0

        markers = self._item_inject_markers.pop(item_id, None)
        if markers:
            for msg in self._msgs:
                if msg.get("role") != "user":
                    continue
                content = msg.get("content")
                if not isinstance(content, str):
                    continue
                for m in markers:
                    if content.startswith(m):
                        msg["content"] = self._SKILL_READ_ELIDED
                        edited += 1
                        break

        ids = self._item_read_ids.pop(item_id, None)
        if ids:
            for msg in self._msgs:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    if (block.get("tool_use_id") or "") not in ids:
                        continue
                    block["content"] = self._SKILL_READ_ELIDED
                    edited += 1

        return edited

    # Match ``[skill auto-injected for p1 v2 (foo)]`` and capture
    # (item_id, plan_version, skill_name) so on_buffer_rebuilt can
    # reconstitute _skill_inject_keys / _item_inject_markers from the
    # messages that survived compaction.
    _SKILL_INJECT_RE = re.compile(
        r"^\[skill auto-injected for ([A-Za-z0-9_]+) v(\d+) \(([^)]+)\)\]",
    )

    def on_buffer_rebuilt(self) -> None:
        """Called after compact/rebuild replaces the buffer.

        ``auto_compact`` carries the most recent rounds forward
        verbatim into the new buffer, so BOTH tracks — auto-injects
        (plain user messages with ``_SKILL_INJECT_PREFIX``) and
        voluntary ``read_file('skills/...')`` tool_results — may
        still live in the new ``_msgs``. We must NOT blindly clear
        either tracking map; previously that caused three bugs:

          * Dedup set (``_skill_inject_keys``) went empty while the
            inject marker still sat in recent messages → next turn
            re-injected the same SKILL.md body.
          * ``_item_inject_markers`` went empty → when the owning
            plan item later settled, ``unload_item_reads`` could not
            find surviving injects to elide.
          * ``_item_read_ids`` was wiped wholesale → voluntary skill
            reads that survived compaction could no longer be
            unloaded at settle time, leaking kB of SKILL.md into the
            buffer past their owning plan item.

        Fix:

          (a) Auto-inject track: rescan ``_msgs`` for inject markers
              and rebuild ``_skill_inject_keys`` / ``_item_inject_markers``
              from what actually survived.
          (b) Voluntary-read track: collect the set of tool_use_ids
              that still appear as tool_result blocks, and drop any
              ``_item_read_ids`` entries referencing ids the rebuild
              threw away. Items whose reads were fully compacted out
              are purged from the map.
        """
        self.__init_read_tracking()

        # (a) Auto-inject markers.
        self._item_inject_markers.clear()
        keys = getattr(self, "_skill_inject_keys", None)
        if keys is None:
            keys = set()
            self._skill_inject_keys = keys
        else:
            keys.clear()
        surviving_tool_use_ids: set[str] = set()
        for msg in self._msgs:
            content = msg.get("content")
            if msg.get("role") == "user" and isinstance(content, str):
                m = self._SKILL_INJECT_RE.match(content)
                if m is not None:
                    item_id, ver_str, skill_name = m.group(1), m.group(2), m.group(3)
                    try:
                        plan_version = int(ver_str)
                    except ValueError:
                        continue
                    keys.add((plan_version, item_id, skill_name))
                    marker = (
                        f"{self._SKILL_INJECT_PREFIX} for {item_id} "
                        f"v{plan_version} ({skill_name})]"
                    )
                    self._item_inject_markers.setdefault(item_id, set()).add(marker)
                continue
            if isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") != "tool_result":
                        continue
                    tid = block.get("tool_use_id")
                    if tid:
                        surviving_tool_use_ids.add(tid)

        # (b) Voluntary-read track: intersect each item's id-set with
        # what the rebuilt buffer still carries; drop empty entries.
        for item_id in list(self._item_read_ids.keys()):
            self._item_read_ids[item_id] &= surviving_tool_use_ids
            if not self._item_read_ids[item_id]:
                del self._item_read_ids[item_id]

    # -- Compression -------------------------------------------------------

    def microcompact(self, min_chars: int = 200, keep_recent: int = 1) -> None:
        """In-place trim of old tool_result content (delegates to compress)."""
        from .compress import microcompact as _microcompact
        _microcompact(self._msgs, min_chars=min_chars, keep_recent=keep_recent)

    async def auto_compact(self, llm, task_dir: str, *, config, tools,
                           feedback=None, last_diagnosis=None,
                           keep_recent_rounds: int = 3,
                           best_metric_str: str = "") -> bool:
        """Run auto_compact. Returns True if buffer was actually compacted.

        compress.auto_compact returns its input list unchanged when
        there isn't enough history to compress (the "no-op signal").
        We detect that via identity check on the internal list and
        return False so the caller can skip post-compact bookkeeping.
        """
        from .compress import auto_compact as _auto_compact
        new_msgs = await _auto_compact(
            self._msgs, llm, task_dir,
            config=config, tools=tools,
            feedback=feedback, last_diagnosis=last_diagnosis,
            keep_recent_rounds=keep_recent_rounds,
            best_metric_str=best_metric_str,
        )
        if new_msgs is self._msgs:
            return False
        self.replace(new_msgs)
        return True

    def force_rebuild(self, task_dir: str, config, feedback=None,
                      last_diagnosis=None,
                      best_metric_str: str = "") -> None:
        """Emergency rebuild — replaces buffer with minimal-context boundary.

        No LLM call. Used when auto_compact fails or PTL recovery is
        needed.
        """
        from .compress import force_rebuild_minimal_context
        new_msgs = force_rebuild_minimal_context(
            task_dir, config,
            feedback=feedback, last_diagnosis=last_diagnosis,
            best_metric_str=best_metric_str,
        )
        self.replace(new_msgs)

    # -- Persistence -------------------------------------------------------

    def save_full_increment(self, task_dir: str, session_dir: str,
                            since_idx: int) -> int:
        """Append messages [since_idx:] to ``messages_full.jsonl``.

        Returns the new total message count, which the caller should
        use as the next ``since_idx`` to keep the cursor advancing.
        """
        from .compress import _save_messages
        new_msgs = self._msgs[since_idx:]
        if new_msgs:
            _save_messages(new_msgs, task_dir, session_dir=session_dir,
                           filename="messages_full.jsonl", mode="a")
        return len(self._msgs)

    def save_latest(self, task_dir: str, session_dir: str) -> None:
        """Overwrite ``messages_latest.jsonl`` with the current buffer state."""
        from .compress import _save_messages
        _save_messages(self._msgs, task_dir, session_dir=session_dir,
                       filename="messages_latest.jsonl", mode="w")

    def load_latest(self, task_dir: str, session_dir: str) -> bool:
        """Load messages from ``messages_latest.jsonl`` into the buffer.

        Returns True on success, False if no file or empty.

        Routes the restored list through ``replace`` so
        ``on_buffer_rebuilt`` rebuilds the skill-tracking maps from
        whatever inject markers and tool_use_ids actually survived
        into the restored buffer. Callers do NOT need to call
        ``on_buffer_rebuilt`` themselves — the invariant is closed
        inside this class.
        """
        from .compress import _load_messages
        restored = _load_messages(task_dir, session_dir=session_dir)
        if restored:
            self.replace(restored)
            return True
        return False
