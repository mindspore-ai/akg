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

"""响应式 Workflow Panel（渲染逻辑收敛在此）。"""

from __future__ import annotations

from typing import Callable, Optional, Any
from rich.text import Text
from textual.widgets import Static
from textual import log

from ai_kernel_generator.cli.cli.ui.state import WorkflowPanelState
from .reactive_panel_base import ReactivePanelBase


class ReactiveWorkflowPanel(ReactivePanelBase[WorkflowPanelState]):
    """响应式 Workflow Panel（渲染逻辑收敛在此）。"""

    def __init__(
        self,
        widget: Optional[Static] = None,
        *,
        empty_placeholder: str = "Waiting for workflow...",
        t_func: Optional[Callable[..., str]] = None,
    ):
        super().__init__(
            widget,
            state_factory=WorkflowPanelState,
            empty_placeholder=empty_placeholder,
        )
        self._t_func: Callable[..., str] = t_func or (lambda k, **_: k)

    def set_translator(self, t_func: Callable[..., str]) -> None:
        self._t_func = t_func
        try:
            self.refresh()
        except Exception as e:
            log.debug(
                "[ReactiveWorkflowPanel] refresh failed after translator set",
                exc_info=e,
            )

    def _render_state(self, state: WorkflowPanelState) -> Any:
        rendered = self._render_workflow(state)
        self._apply_title(self._title_for_workflow(state))
        return rendered

    def _apply_title(self, title: str) -> None:
        if not self._widget:
            return
        try:
            self._widget.border_title = str(title or "")
            try:
                self._widget.refresh()
            except Exception as e:
                log.debug("[ReactiveWorkflowPanel] widget.refresh failed", exc_info=e)
        except Exception as e:
            log.warning("[ReactiveWorkflowPanel] apply title failed", exc_info=e)
            return

    def _title_for_workflow(self, state: WorkflowPanelState) -> str:
        base = str(self._t_func("tui.title.workflow") or "workflow")
        node = (state.current_node or "").strip()
        if not node:
            return base
        run_no = 0
        try:
            run_no = int(state.current_node_run_no or 0)
        except (TypeError, ValueError) as e:
            log.debug(
                "[ReactiveWorkflowPanel] current_node_run_no cast failed", exc_info=e
            )
            run_no = 0
        suffix = f"#{run_no}" if run_no else ""
        title = f"{base} | {node}{suffix}"
        return title

    def _fmt_node(self, name: str, state: WorkflowPanelState) -> Text:
        from ai_kernel_generator.cli.cli.constants import DisplayStyle

        n = (name or "").strip() or "-"
        cnt = 0
        try:
            cnt = int((state.node_run_counts or {}).get(n, 0) or 0)
        except (TypeError, ValueError) as e:
            log.debug("[ReactiveWorkflowPanel] node_run_counts cast failed", exc_info=e)
            cnt = 0

        if (state.current_node or "").strip() and n == (
            state.current_node or ""
        ).strip():
            run_no = 0
            try:
                run_no = int(state.current_node_run_no or 0) or cnt
            except (TypeError, ValueError) as e:
                log.debug(
                    "[ReactiveWorkflowPanel] current_node_run_no cast failed; fallback cnt",
                    exc_info=e,
                )
                run_no = cnt
            suffix = f"#{run_no}" if run_no else ""
            return Text(f"{n}{suffix}", style=DisplayStyle.BOLD_YELLOW)

        suffix = f"×{cnt}" if cnt else ""
        return Text(f"{n}{suffix}", style=DisplayStyle.WHITE)

    def _workflow_graph_lines(self, state: WorkflowPanelState) -> list[Text]:
        from ai_kernel_generator.cli.cli.constants import DisplayStyle

        wf = (state.workflow_name or "").lower()

        def _join_nodes(nodes: list[Text]) -> Text:
            line = Text()
            for i, node in enumerate(nodes):
                if i:
                    line.append(" ──▶ ")
                line.append_text(node)
            return line

        def _node_label(name: str) -> Text:
            n = (name or "").strip() or "-"
            try:
                cnt = int((state.node_run_counts or {}).get(n, 0) or 0)
            except Exception:
                cnt = 0
            label = f"{n}×{cnt}"
            style = DisplayStyle.WHITE
            if (state.current_node or "").strip() == n:
                style = DisplayStyle.BOLD_YELLOW
            return Text(label, style=style)

        def _graph_fixed(node_names: list[str]) -> list[Text]:
            # 固定图：上方 ok path（线性），失败从 verifier 下探到 conductor，然后回到 coder（闭环箭头回到同一处）
            nodes = [_node_label(n) for n in node_names]
            line1 = _join_nodes(nodes + [Text("END(ok)", style=DisplayStyle.DIM)])

            # 计算 coder/verifier 的中心列，保证连接线稳定对齐
            starts: list[int] = []
            cursor = 0
            for i, node in enumerate(nodes + [Text("END(ok)")]):
                starts.append(cursor)
                cursor += len(node.plain)
                if i != len(nodes):
                    cursor += len(" ──▶ ")

            def _center(idx: int) -> int:
                return starts[idx] + max(0, len(nodes[idx].plain) // 2)

            try:
                verifier_idx = node_names.index("verifier")
            except ValueError:
                return [line1]
            try:
                coder_idx = node_names.index("coder")
            except ValueError:
                coder_idx = 0

            coder_col = _center(coder_idx)
            verifier_col = _center(verifier_idx)

            conductor = _node_label("conductor")
            fail_prefix = "└──(fail)──▶ "

            line2 = Text(
                (" " * coder_col)
                + "▲"
                + (" " * max(0, verifier_col - coder_col - 1))
                + "│",
                style=DisplayStyle.DIM,
            )
            line3 = Text.assemble(
                (
                    (" " * coder_col)
                    + "│"
                    + (" " * max(0, verifier_col - coder_col - 1))
                    + fail_prefix,
                    DisplayStyle.DIM,
                ),
                conductor,
                (" ──┐", DisplayStyle.DIM),
            )
            hook_col = len(line3.plain) - 1
            line4 = Text(
                (" " * coder_col)
                + "◀"
                + ("─" * max(0, hook_col - coder_col - 1))
                + "┘",
                style=DisplayStyle.DIM,
            )
            return [line1, line2, line3, line4]

        # 图严格按 workflow 固定；不再依据 seen_nodes 决定拓扑
        if "coder_only_workflow" in wf or "coder_only" in wf:
            # 你期望的固定流程：designer -> coder -> verifier -> ok -> sketch
            return _graph_fixed(["designer", "coder", "verifier", "sketch"])
        if "default_workflow" in wf or "default" in wf:
            return _graph_fixed(["designer", "coder", "verifier"])

        if "verifier_only" in wf:
            verifier = self._fmt_node("verifier", state)
            return [_join_nodes([verifier, Text("END", style=DisplayStyle.DIM)])]

        # 兜底：线性展示已出现的节点
        nodes: list[str] = []
        for n in state.seen_nodes or []:
            if n and n not in nodes:
                nodes.append(n)
        if not nodes:
            return [
                Text(
                    str(self._t_func("presenter.wait_node_events")),
                    style=DisplayStyle.DIM,
                )
            ]

        return [_join_nodes([self._fmt_node(n, state) for n in nodes])]

    def _render_workflow(self, state: WorkflowPanelState) -> Text:
        from ai_kernel_generator.cli.cli.constants import DisplayStyle

        node = (state.current_node or "").strip() or "-"
        st = (state.current_node_status or "").strip().lower()

        out = Text()

        # 当前节点行
        out.append(
            str(self._t_func("presenter.label.current")), style=DisplayStyle.CYAN
        )
        out.append(": ")
        out.append_text(self._fmt_node(node, state))
        if st == "running":
            out.append("  ")
            out.append(
                str(self._t_func("presenter.status.running")),
                style=DisplayStyle.YELLOW,
            )
        elif st == "done":
            out.append("  ")
            out.append(
                str(self._t_func("presenter.status.done")),
                style=DisplayStyle.GREEN,
            )

        # 图
        out.append("\n\n")
        out.append(str(self._t_func("presenter.label.graph")), style=DisplayStyle.DIM)
        out.append(":\n")
        glines = self._workflow_graph_lines(state)
        for i, line in enumerate(glines):
            out.append_text(line)
            if i != len(glines) - 1:
                out.append("\n")

        # 进度（可选）
        if state.progress:
            out.append("\n\n")
            out.append(
                str(self._t_func("presenter.label.progress")), style=DisplayStyle.DIM
            )
            out.append(":\n")
            prog_lines = self._render_progress_lines(state)
            for i, line in enumerate(prog_lines):
                out.append_text(line)
                if i != len(prog_lines) - 1:
                    out.append("\n")
        else:
            # workflow 刚启动但尚未收到结构化进度：展示一个"如何切换 watch"的提示
            try:
                hint = str(self._t_func("presenter.hint.switch_watch") or "").strip()
            except Exception as e:
                log.debug(
                    "[ReactiveWorkflowPanel] read hint.switch_watch failed", exc_info=e
                )
                hint = ""
            # 已结束（done）时不应再显示“初始/提示”文案，避免误导用户以为进度未刷新
            if (
                hint
                and (state.workflow_name or state.current_node or state.seen_nodes)
                and (state.current_node_status or "").strip().lower() != "done"
            ):
                out.append("\n\n")
                out.append(
                    str(self._t_func("presenter.label.progress")),
                    style=DisplayStyle.DIM,
                )
                out.append(":\n")
                out.append_text(Text(hint, style=DisplayStyle.DIM))

        return out

    def _render_progress_lines(self, state: WorkflowPanelState) -> list[Text]:
        from ai_kernel_generator.cli.cli.constants import DisplayStyle

        d = dict(state.progress or {})
        if not d:
            return []

        round_idx = int(d.get("round", 0) or 0)
        max_rounds = int(d.get("max_rounds", 0) or 0)
        total = int(d.get("total", 0) or 0)
        done = int(d.get("done", 0) or 0)
        ok = int(d.get("ok", 0) or 0)
        fail = int(d.get("fail", 0) or 0)
        tasks = d.get("tasks") if isinstance(d.get("tasks"), dict) else {}

        header = str(
            self._t_func(
                "presenter.evolve.progress_header",
                round=round_idx,
                max_rounds=max_rounds,
                done=done,
                total=total,
                ok=ok,
                fail=fail,
            )
        )

        def _bar(status: str) -> str:
            if status == "queued":
                return "----------"
            if status == "running":
                return "#####-----"
            if status in ["done", "fail"]:
                return "##########"
            return "----------"

        watch_tid = (state.watch_task_id or "").strip()
        lines: list[Text] = [Text(header, style=DisplayStyle.BOLD)]
        for task_name, status in sorted(tasks.items()):
            st = str(status)
            if st == "done":
                st_style = DisplayStyle.BOLD_GREEN
                st_label = "presenter.evolve.status.done"
            elif st == "fail":
                st_style = DisplayStyle.BOLD_RED
                st_label = "presenter.evolve.status.fail"
            elif st == "running":
                st_style = DisplayStyle.YELLOW
                st_label = "presenter.evolve.status.run"
            else:
                st_style = DisplayStyle.DIM
                st_label = "presenter.evolve.status.queue"

            row = Text()
            row.append(_bar(st))
            row.append(" ")
            row.append(str(self._t_func(st_label)), style=st_style)
            row.append(" ")
            row.append(str(task_name))

            if watch_tid and str(task_name) == watch_tid:
                line = Text()
                line.append("▶", style=DisplayStyle.BOLD_CYAN)
                line.append(" ")
                line.append_text(row)
                lines.append(line)
            else:
                line = Text("  ")
                line.append_text(row)
                lines.append(line)

        return lines
