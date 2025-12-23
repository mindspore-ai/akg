# Copyright 2025 Huawei Technologies Co., Ltd
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

from __future__ import annotations

import os
from queue import Empty
from typing import TYPE_CHECKING, Callable

from textual import log
from textual.widgets import Tab

from .widgets import TraceListItem
from ai_kernel_generator.cli.cli.ui.commands import (
    AppendTrace,
    ClearTrace,
    Focus,
    PatchInfoState,
    PatchWorkflowState,
    Quit,
    SetActiveTaskTab,
    SetInput,
    SetTaskTabs,
    SetTrace,
    SetTraceTitle,
    UICommand,
    UpdateInfoState,
    UpdateWorkflowState,
)
from ai_kernel_generator.cli.cli.ui.state import InfoPanelState, WorkflowPanelState
from ai_kernel_generator.cli.cli.ui.types import (
    ClearMainContent,
    MainContent,
    SyntaxBlockMainContent,
    TraceAnchorMainContent,
)
from ai_kernel_generator.cli.cli.utils.i18n import t

if TYPE_CHECKING:
    from .app import SplitViewApp


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() in ["1", "true", "yes", "on", "y"]


# 输出队列可能非常高频（流式输出时会刷屏）；默认只记录关键事件，避免 textual.log 爆量。
_LOG_OUTPUT_VERBOSE: bool = _env_bool("AIKG_TUI_LOG_OUTPUT_VERBOSE", False)


class _QueueProcessorBase:
    def __init__(self, app: "SplitViewApp") -> None:
        self.app = app

    def _call_later(self, func: Callable[[], None]) -> None:
        try:
            self.app.call_later(func)
        except Exception as e:
            log.warning(
                "[QueueProcessor] app.call_later failed; fallback direct call",
                exc_info=e,
            )
            func()


class OutputQueueProcessor(_QueueProcessorBase):
    """负责处理 worker->UI 的 output_queue。"""

    def _snapshot_chat_scroll(self) -> tuple[float, int] | None:
        if self.app.chat_log is None:
            return None
        try:
            scroll_y = float(self.app.chat_log.scroll_y)
        except Exception as e:
            log.debug("[OutputQueue] read scroll_y failed; skip snapshot", exc_info=e)
            return None
        try:
            start_line = int(getattr(self.app.chat_log, "_start_line", 0) or 0)
        except (TypeError, ValueError) as e:
            log.debug("[OutputQueue] read _start_line failed; fallback 0", exc_info=e)
            start_line = 0
        return (scroll_y, start_line)

    def _restore_chat_scroll(self, snapshot: tuple[float, int] | None) -> None:
        if snapshot is None or self.app.chat_log is None:
            return
        scroll_y_before, start_line_before = snapshot
        try:
            start_line_after = int(getattr(self.app.chat_log, "_start_line", 0) or 0)
        except (TypeError, ValueError) as e:
            log.debug("[OutputQueue] read _start_line failed; keep snapshot", exc_info=e)
            start_line_after = start_line_before
        trimmed = max(0, start_line_after - start_line_before)
        target_y = float(scroll_y_before - trimmed)
        if target_y < 0:
            target_y = 0.0
        try:
            max_y = float(self.app.chat_log.max_scroll_y)
        except Exception:
            max_y = None
        if max_y is not None and target_y > max_y:
            target_y = max_y
        try:
            current_y = float(self.app.chat_log.scroll_y)
        except Exception:
            current_y = target_y
        if abs(current_y - target_y) < 0.5:
            return
        try:
            # Keep the view stable when tail is off, even if lines are trimmed.
            self.app.chat_log.scroll_to(y=target_y, animate=False, immediate=True)
        except Exception as e:
            log.debug("[OutputQueue] restore scroll failed", exc_info=e)

    def drain(self) -> None:
        wrote_output = False
        was_at_bottom = self._was_chat_at_bottom()
        scroll_snapshot = self._snapshot_chat_scroll()
        drained = 0
        try:
            while True:
                content = self.app.output_queue.get_nowait()
                drained += 1
                if _LOG_OUTPUT_VERBOSE:
                    log.debug(
                        "[OutputQueue] content",
                        type=type(content).__name__,
                    )
                if self._handle_content(content):
                    wrote_output = True
        except Empty:
            pass

        if wrote_output and self.app.chat_log is not None:
            should_auto_scroll = bool(self.app.follow_tail and was_at_bottom)
            if should_auto_scroll:
                try:
                    self.app.chat_log.scroll_end(animate=False)
                    if _LOG_OUTPUT_VERBOSE:
                        log.debug("[OutputQueue] auto_scroll_end")
                except Exception as e:
                    log.debug("[OutputQueue] auto_scroll_end failed", exc_info=e)
            else:
                self._restore_chat_scroll(scroll_snapshot)
        if drained and not _LOG_OUTPUT_VERBOSE:
            log.debug(
                "[OutputQueue] drained",
                n=drained,
                wrote_output=bool(wrote_output),
            )

    def _was_chat_at_bottom(self) -> bool:
        if self.app.chat_log is None:
            return True
        try:
            return bool(self.app.chat_log.is_vertical_scroll_end)
        except Exception as e:
            log.debug(
                "[OutputQueue] is_vertical_scroll_end failed; fallback scroll_y",
                exc_info=e,
            )
            try:
                return self.app.chat_log.scroll_y >= self.app.chat_log.max_scroll_y
            except Exception as e2:
                log.debug(
                    "[OutputQueue] scroll_y fallback failed; assume bottom", exc_info=e2
                )
                return True

    def _handle_content(self, content: MainContent) -> bool:
        if isinstance(content, ClearMainContent):
            log("[OutputQueue] Clearing chat content")
            self.app.chat.clear()
            return False

        if isinstance(content, TraceAnchorMainContent):
            log.debug(
                "[OutputQueue] trace_anchor",
                task_id=str(content.task_id or ""),
                event_idx=int(content.event_idx or 0),
            )
            task_id_s = str(content.task_id or "")
            try:
                event_idx_i = int(content.event_idx or 0)
            except (TypeError, ValueError) as e:
                log.debug(
                    "[OutputQueue] event_idx cast failed; fallback 0", exc_info=e
                )
                event_idx_i = 0
            try:
                self.app.chat.record_trace_anchor(
                    task_id_s, event_idx_i
                )
            except Exception as e:
                log.warning("[OutputQueue] record_trace_anchor failed", exc_info=e)
            try:
                anchor_y = int(
                    self.app.trace_anchors.get((task_id_s, event_idx_i), 0) or 0
                )
            except (TypeError, ValueError) as e:
                log.debug("[OutputQueue] anchor_y cast failed; fallback 0", exc_info=e)
                anchor_y = 0
            try:
                if self.app.trace_panel is not None and hasattr(
                    self.app.trace_panel, "update_anchor"
                ):
                    self.app.trace_panel.update_anchor(
                        task_id_s, event_idx_i, anchor_y
                    )
            except Exception as e:
                log.debug("[OutputQueue] update_anchor failed", exc_info=e)
            # 不写入 chat
            return False

        if isinstance(content, SyntaxBlockMainContent):
            code = str(content.code or "")
            lexer_name = str(content.lexer_name or "text")
            add_end_separator = bool(content.add_end_separator)
            line_number_start = (
                int(content.line_number_start)
                if content.line_number_start is not None
                else None
            )
            log.info(
                "[OutputQueue] syntax_block",
                lexer=lexer_name,
                lines=len(code.splitlines()),
                add_end_separator=bool(add_end_separator),
                line_number_start=line_number_start
                if line_number_start is not None
                else "",
            )
            self.app.chat.write_syntax_block(
                code,
                lexer_name,
                add_end_separator=add_end_separator,
                line_number_start=line_number_start,
            )
            return True

        if isinstance(content, str):
            if _LOG_OUTPUT_VERBOSE:
                log.debug(
                    "[OutputQueue] markup",
                    preview=(content or "")[:80],
                )
            self.app.chat.write_markup(content)
            return True

        if _LOG_OUTPUT_VERBOSE:
            log.debug(
                "[OutputQueue] renderable",
                type=type(content).__name__,
            )
        self.app.chat.write_renderable(content)
        return True


class CommandQueueProcessor(_QueueProcessorBase):
    """负责处理 worker->UI 的 command_queue。"""

    def __init__(self, app: "SplitViewApp") -> None:
        super().__init__(app)

    def drain(self) -> None:
        try:
            while True:
                command = self.app.command_queue.get_nowait()
                log(f"[CommandQueue] Processing command: {type(command).__name__}")
                self._handle_command(command)
        except Empty:
            pass

    def _handle_command(self, command: UICommand) -> None:
        if isinstance(command, Quit):
            log("[CommandQueue] Quit command received")
            self.app.exit()
            return

        try:
            if isinstance(command, SetInput):
                self._cmd_set_input(command.enabled)
                return
            if isinstance(command, Focus):
                self._cmd_focus(command.target)
                return
            if isinstance(command, UpdateInfoState):
                self._cmd_update_info_state(command.state)
                return
            if isinstance(command, PatchInfoState):
                self._cmd_patch_info_state(command.patch)
                return
            if isinstance(command, UpdateWorkflowState):
                self._cmd_update_workflow_state(command.state)
                return
            if isinstance(command, PatchWorkflowState):
                self._cmd_patch_workflow_state(command.patch)
                return
            if isinstance(command, AppendTrace):
                self._cmd_append_trace(command.text, command.task_id, command.event_idx)
                return
            if isinstance(command, ClearTrace):
                self._cmd_clear_trace()
                return
            if isinstance(command, SetTraceTitle):
                self._cmd_set_trace_title(command.title)
                return
            if isinstance(command, SetTrace):
                self._cmd_set_trace(command.items)
                return
            if isinstance(command, SetTaskTabs):
                self._cmd_set_task_tabs(command.items, command.active_task_id)
                return
            if isinstance(command, SetActiveTaskTab):
                self._cmd_set_task_tab_active(command.task_id)
                return
        except Exception as e:
            log(f"[CommandQueue] Failed to handle UICommand: {command!r}", exc_info=e)
            return

        log(f"[CommandQueue] Unhandled UICommand type: {type(command).__name__}")

    def _cmd_set_input(self, enabled: bool) -> None:
        if self.app.user_input is None:
            return

        enabled_b = bool(enabled)
        log(f"[CommandQueue] SetInput: enabled={enabled_b}")
        self.app._input_enabled = enabled_b
        self.app.user_input.disabled = not enabled_b
        self.app.user_input.placeholder = (
            t("tui.placeholder.input_enabled_hint")
            if enabled_b
            else t("tui.placeholder.input_disabled")
        )
        if enabled_b:
            try:
                self.app.user_input.focus()
            except Exception as e:
                log.debug("[CommandQueue] user_input.focus failed", exc_info=e)

    def _cmd_focus(self, target: str) -> None:
        target_n = str(target or "").strip().lower()
        log(f"[CommandQueue] Focus: target={target_n}")
        try:
            if target_n in ["chat", "main", "left"] and self.app.chat_log is not None:
                self.app.chat_log.focus()
            elif (
                target_n in ["task", "right", "workflow"]
                and self.app.workflow_panel is not None
            ):
                self.app.workflow_panel.focus()
            elif target_n in ["trace"] and self.app.trace_panel is not None:
                self.app.trace_panel.focus()
            elif (
                target_n in ["input"]
                and self.app.user_input is not None
                and not self.app.user_input.disabled
            ):
                self.app.user_input.focus()
        except Exception as e:
            log.warning("[CommandQueue] focus failed", target=target_n, exc_info=e)

    def _cmd_update_info_state(self, state: InfoPanelState) -> None:
        def _do_update_state() -> None:
            try:
                self.app.reactive_info_panel.update(state)
            except Exception as e:
                log.warning(
                    "[CommandQueue] reactive_info_panel.update failed", exc_info=e
                )

        self._call_later(_do_update_state)

    def _cmd_patch_info_state(self, kwargs: dict[str, object]) -> None:
        def _do_patch_state() -> None:
            try:
                self.app.reactive_info_panel.patch(**kwargs)
            except Exception as e:
                log.warning(
                    "[CommandQueue] reactive_info_panel.patch failed", exc_info=e
                )

        self._call_later(_do_patch_state)

    def _cmd_update_workflow_state(self, state: WorkflowPanelState) -> None:
        def _do_update_state() -> None:
            try:
                log(
                    f"[CommandQueue] UpdateWorkflowState: current_node={state.current_node}, status={state.current_node_status}"
                )
                self.app.reactive_workflow_panel.update(state)
            except Exception as e:
                log.warning(
                    "[CommandQueue] reactive_workflow_panel.update failed", exc_info=e
                )

        self._call_later(_do_update_state)

        return

    def _cmd_patch_workflow_state(self, kwargs: dict[str, object]) -> None:
        def _do_patch_state() -> None:
            try:
                self.app.reactive_workflow_panel.patch(**kwargs)
            except Exception as e:
                log.warning(
                    "[CommandQueue] reactive_workflow_panel.patch failed", exc_info=e
                )

        self._call_later(_do_patch_state)

        return

    def _cmd_append_trace(self, text: str, task_id: str, event_idx: int = 0) -> None:
        text_s = str(text or "")
        task_id_s = str(task_id or "")
        try:
            event_idx_i = int(event_idx or 0)
        except (TypeError, ValueError) as e:
            log.debug("[CommandQueue] event_idx cast failed; fallback 0", exc_info=e)
            event_idx_i = 0

        log(
            f"[CommandQueue] AppendTrace: task_id={task_id_s}, event_idx={event_idx_i}, text={text_s[:50]}..."
        )

        # 优先使用 output_queue 里记录的精确锚点（写入前的 virtual y）
        try:
            anchor_y = int(self.app.trace_anchors.get((task_id_s, event_idx_i), 0) or 0)
        except (TypeError, ValueError) as e:
            log.debug("[CommandQueue] anchor_y cast failed; fallback 0", exc_info=e)
            anchor_y = 0

        def _do_append() -> None:
            if self.app.trace_panel is None:
                return
            try:
                self.app.trace_panel.append(
                    TraceListItem(
                        text_s,
                        task_id=task_id_s,
                        event_idx=event_idx_i,
                        anchor_y=anchor_y,
                    )
                )
                try:
                    self.app.trace_panel.scroll_end(animate=False)
                except Exception as e:
                    log.debug(
                        "[CommandQueue] trace_panel.scroll_end failed", exc_info=e
                    )
            except Exception as e:
                log.warning("[CommandQueue] trace_panel.append failed", exc_info=e)
                return

        self._call_later(_do_append)

    def _cmd_clear_trace(self) -> None:
        def _do_clear() -> None:
            if self.app.trace_panel is None:
                return
            try:
                self.app.trace_panel.clear()
            except Exception as e:
                log.warning("[CommandQueue] trace_panel.clear failed", exc_info=e)

        self._call_later(_do_clear)

    def _cmd_set_trace_title(self, title: str) -> None:
        title_s = str(title or "")
        if self.app.trace_panel is None or not title_s:
            return

        def _do_title() -> None:
            if self.app.trace_panel is None:
                return
            try:
                self.app.trace_panel.border_title = title_s
                try:
                    self.app.trace_panel.refresh()
                except Exception as e:
                    log.debug("[CommandQueue] trace_panel.refresh failed", exc_info=e)
            except Exception as e:
                log.warning(
                    "[CommandQueue] set trace_panel.border_title failed", exc_info=e
                )

        self._call_later(_do_title)

    def _cmd_set_trace(self, items: list[tuple[str, str, int]]) -> None:
        def _do_set() -> None:
            if self.app.trace_panel is None:
                return
            try:
                self.app.trace_panel.clear()
            except Exception as e:
                log.warning("[CommandQueue] trace_panel.clear failed", exc_info=e)
            try:
                if isinstance(items, list):
                    for it in items:
                        try:
                            text, task_id, event_idx = it
                        except (TypeError, ValueError) as e:
                            log.debug(
                                "[CommandQueue] trace item unpack failed; skip",
                                exc_info=e,
                            )
                            continue
                        try:
                            anchor_y = int(
                                self.app.trace_anchors.get(
                                    (str(task_id), int(event_idx or 0)), 0
                                )
                                or 0
                            )
                        except (TypeError, ValueError) as e:
                            log.debug(
                                "[CommandQueue] anchor_y cast failed; fallback 0",
                                exc_info=e,
                            )
                            anchor_y = 0
                        try:
                            self.app.trace_panel.append(
                                TraceListItem(
                                    str(text),
                                    task_id=str(task_id),
                                    event_idx=int(event_idx or 0),
                                    anchor_y=anchor_y,
                                )
                            )
                        except Exception as e:
                            log.warning(
                                "[CommandQueue] trace_panel.append failed; skip item",
                                exc_info=e,
                            )
                            continue
                try:
                    self.app.trace_panel.scroll_end(animate=False)
                except Exception as e:
                    log.debug(
                        "[CommandQueue] trace_panel.scroll_end failed", exc_info=e
                    )
            except Exception as e:
                log.warning("[CommandQueue] set trace failed", exc_info=e)
                return

        self._call_later(_do_set)

    @staticmethod
    def _safe_tab_id(task_id: str) -> str:
        # Textual Widget id 不能以数字开头：统一加前缀
        raw = str(task_id or "")
        safe = []
        for ch in raw:
            if ch.isalnum() or ch in ["_", "-"]:
                safe.append(ch)
            else:
                safe.append("_")
        return "tid_" + "".join(safe)

    def _cmd_set_task_tabs(
        self, items: list[tuple[str, str]], active: str = ""
    ) -> None:
        def _do_set_tabs() -> None:
            if self.app.task_tabs is None:
                return
            # 如果是空更新，不要动现有 Tabs（避免"开始有 task_init，后来啥都没了"）
            if not items:
                log("[CommandQueue] SetTaskTabs: Empty items, skipping")
                return

            # 先把输入 items 规范化；若规范化后为空，则不要动现有 Tabs
            normalized: list[tuple[str, str]] = []
            try:
                if isinstance(items, list):
                    for it in items:
                        try:
                            tid, label = it
                        except (TypeError, ValueError) as e:
                            log.debug(
                                "[CommandQueue] tab item unpack failed; skip",
                                exc_info=e,
                            )
                            continue
                        tid_s = str(tid or "").strip()
                        if not tid_s:
                            continue
                        lab_s = str(label or tid_s).strip()
                        if not lab_s:
                            lab_s = tid_s
                        normalized.append((tid_s, lab_s))
            except Exception as e:
                log.warning("[CommandQueue] normalize tabs failed", exc_info=e)
                normalized = []
            if not normalized:
                log("[CommandQueue] SetTaskTabs: No valid items after normalization")
                return

            log(f"[CommandQueue] SetTaskTabs: {len(normalized)} tabs, active={active}")

            # 关键修复：不要 clear() 再重建（clear/remove 是异步的，未 await 会在稍后把新 tab 又删掉，表现为"标题消失/变空"）
            try:
                for tid_s, lab_s in normalized:
                    tab_id = self._safe_tab_id(tid_s)
                    self.app._tab_id_to_task_id[tab_id] = tid_s
                    self.app._task_id_to_tab_id[tid_s] = tab_id

                    existing = None
                    try:
                        # 注意：Tab 实际挂在 Tabs 的内部容器里（不是 immediate child），
                        # get_child_by_id() 找不到会导致每次刷新都重复 add_tab，最终出现 tab 栏"越刷越乱"。
                        existing = self.app.task_tabs.query_one(f"#{tab_id}", Tab)
                    except Exception as e:
                        log.debug(
                            "[CommandQueue] query_one tab failed",
                            tab_id=tab_id,
                            exc_info=e,
                        )
                        existing = None

                    if isinstance(existing, Tab):
                        try:
                            existing.label = lab_s
                            log(
                                f"[CommandQueue] SetTaskTabs: Updated existing tab {tab_id} -> {lab_s}"
                            )
                        except Exception as e:
                            log.warning(
                                "[CommandQueue] update existing tab failed",
                                tab_id=tab_id,
                                exc_info=e,
                            )
                    else:
                        try:
                            self.app.task_tabs.add_tab(Tab(lab_s, id=tab_id))
                            log(
                                f"[CommandQueue] SetTaskTabs: Added new tab {tab_id} -> {lab_s}"
                            )
                        except Exception as e:
                            # 不能 fallback 到 add_tab(label)：会生成 auto id（tab-*），
                            # 后续刷新无法命中/复用，导致重复 tab 与点击无效。
                            log.warning(
                                "[CommandQueue] add_tab failed",
                                tab_id=tab_id,
                                exc_info=e,
                            )

                # 设置 active（Tabs.active 期望 tab id）
                if active:
                    tab_id = self.app._task_id_to_tab_id.get(active, "")
                    if tab_id:
                        try:
                            self.app._suppress_tab_event = True
                            self.app.task_tabs.active = tab_id
                            self.app._active_task_id = active
                            log(
                                f"[CommandQueue] SetTaskTabs: Set active tab to {tab_id}"
                            )
                        except Exception as e:
                            log.warning(
                                "[CommandQueue] set active tab failed",
                                tab_id=tab_id,
                                exc_info=e,
                            )
                        finally:
                            try:
                                self.app.call_later(
                                    lambda: setattr(
                                        self.app, "_suppress_tab_event", False
                                    )
                                )
                            except Exception as e:
                                log.warning(
                                    "[CommandQueue] clear suppress_tab_event failed",
                                    exc_info=e,
                                )
                                self.app._suppress_tab_event = False
            except Exception as e:
                log.warning("[CommandQueue] SetTaskTabs failed", exc_info=e)
                return

        self._call_later(_do_set_tabs)

    def _cmd_set_task_tab_active(self, task_id: str) -> None:
        task_id_s = str(task_id or "").strip()
        if self.app.task_tabs is None or not task_id_s:
            return

        log(f"[CommandQueue] SetActiveTaskTab: task_id={task_id_s}")

        def _do_set_active() -> None:
            if self.app.task_tabs is None:
                return
            tab_id = self.app._task_id_to_tab_id.get(task_id_s, "")
            if not tab_id:
                log(
                    f"[CommandQueue] SetActiveTaskTab: tab_id not found for task_id={task_id_s}"
                )
                return
            try:
                self.app._suppress_tab_event = True
                self.app.task_tabs.active = tab_id
                self.app._active_task_id = task_id_s
                log(f"[CommandQueue] SetActiveTaskTab: Activated tab {tab_id}")
                try:
                    self.app.task_tabs.refresh()
                except Exception as e:
                    log.debug("[CommandQueue] task_tabs.refresh failed", exc_info=e)
            except Exception as e:
                log.warning(
                    "[CommandQueue] set_active_task_tab failed",
                    tab_id=tab_id,
                    exc_info=e,
                )
            finally:
                try:
                    self.app.call_later(
                        lambda: setattr(self.app, "_suppress_tab_event", False)
                    )
                except Exception as e:
                    log.warning(
                        "[CommandQueue] clear suppress_tab_event failed", exc_info=e
                    )
                    self.app._suppress_tab_event = False

        self._call_later(_do_set_active)


class SplitViewQueueProcessor:
    """SplitViewApp 的队列处理器（output_queue + command_queue）。"""

    def __init__(self, app: "SplitViewApp") -> None:
        self.app = app
        self.output = OutputQueueProcessor(app)
        self.command = CommandQueueProcessor(app)

    def process(self) -> None:
        self.output.drain()
        self.command.drain()
