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
from textual import log as textual_log  # type: ignore

import logging
import os
import traceback
from typing import TYPE_CHECKING, Any, Optional

from rich.console import Console as RichConsole
from rich.text import Text

from ai_kernel_generator.cli.cli.constants import (
    DisplayStyle,
    NodeName,
    SyntaxLanguage,
    UISymbol,
)
from ai_kernel_generator.cli.cli.presenter.stream import StreamRenderer
from ..ui.types import SyntaxBlockMainContent, TraceAnchorMainContent
from ..utils.i18n import t
from ...messages import (
    LLMEndMessage,
    LLMStartMessage,
    LLMStreamMessage,
    NodeEndMessage,
    NodeStartMessage,
    ProgressMessage,
)

if TYPE_CHECKING:
    from .core import CLIPresenter


logger = logging.getLogger(__name__)

# Textual devtools 里更稳定的日志通道（会出现在 Textual Development Console）


class _PassiveLayoutProxy:
    """
    给后台 StreamRenderer 使用的“哑布局”：不把内容写入当前 UI，但保留宽度查询能力，
    以保证后台渲染出的换行/行号与前台一致。
    """

    def __init__(self, layout_manager: Any):
        self._lm = layout_manager

    def update_main_content(self, content: Any) -> None:  # noqa: ARG002
        return

    def get_panel_content_width(self, panel: str = "chat") -> Optional[int]:
        try:
            if hasattr(self._lm, "get_panel_content_width"):
                return self._lm.get_panel_content_width(panel)
        except Exception as e:
            textual_log.debug(
                "[Presenter] get_panel_content_width failed; fallback None",
                panel=str(panel or ""),
                exc_info=e,
            )
        return None


def _llm_language_from_agent(agent: str) -> str:
    if agent in [NodeName.CODER, NodeName.TASK_INIT]:
        return SyntaxLanguage.PYTHON
    return SyntaxLanguage.TEXT


def _normalize_boolish(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ["true", "1", "yes", "y", "pass", "ok", "success"]:
            return True
        if s in ["false", "0", "no", "n", "fail", "error", "failed"]:
            return False
        return None
    return bool(value)


def _format_token_info(
    *, prompt_tokens: Any, reasoning_tokens: Any, output_tokens: Any, total_tokens: Any
) -> str:
    token_info: list[str] = []
    if prompt_tokens is not None:
        token_info.append(f"input {prompt_tokens}")
    if reasoning_tokens is not None:
        token_info.append(f"reasoning {reasoning_tokens}")
    if output_tokens is not None:
        token_info.append(f"output {output_tokens}")
    if total_tokens is not None:
        token_info.append(f"total {total_tokens}")
    return f" | Tokens: {', '.join(token_info)}" if token_info else ""


class PresenterEventHandlers:
    """把核心事件处理从 CLIPresenter 中拆出来，降低 core.py 体积。"""

    def __init__(self, presenter: "CLIPresenter") -> None:
        self._p = presenter
        # agent -> 最近一次 llm_start 对应的 task_id（用于 llm_stream 未带 task_id 时做归属）
        self._llm_agent_task: dict[str, str] = {}
        # main tab 提示已覆盖的 task_id（避免刷屏）
        self._main_tab_hint_tasks: set[str] = set()

    def render_task_stream_buffer(self, task_id: str) -> None:
        """
        切换到某个 task 时：把该 task 已累计的 llm_stream buffer 一次性渲染到当前 UI，
        避免后台逐 chunk 渲染带来的巨大性能开销。
        """
        p = self._p
        tid = str(task_id or "").strip()
        if not tid:
            return
        if not p.use_stream:
            return
        if not self._should_render_task(tid):
            return
        try:
            sess = p.tasks.stream_session(tid)
        except Exception as e:
            textual_log.warning(
                "[Presenter] get stream_session failed", task_id=tid, exc_info=e
            )
            return

        try:
            snap = sess.buffer_debug_snapshot()
            if snap.get("chunks") or snap.get("buf_len"):
                textual_log.info(
                    "[Presenter] render_task_stream_buffer", task_id=tid, **snap
                )
        except Exception as e:
            textual_log.debug(
                "[Presenter] stream buffer debug snapshot failed; ignore",
                task_id=tid,
                exc_info=e,
            )

        try:
            sess.flush_buffer(active=True)
        except Exception as e:
            textual_log.warning(
                "[Presenter] flush stream buffer failed", task_id=tid, exc_info=e
            )

    def _cache_llm_response_for_replay(
        self,
        task_id: Optional[str],
        *,
        agent: str,
        model: str,
        language: str,
        response: str,
    ) -> None:
        """
        非 active task：不做实时流式渲染，但需要保证“切换 tab 后能看到完整内容”。
        这里在 llm_end 阶段用一个离屏 StreamRenderer 把完整 response 渲染成 MainContent 序列并写入缓存。
        """
        p = self._p
        tid = str(task_id or "").strip()
        if (not tid) or (not response):
            return
        try:
            proxy = _PassiveLayoutProxy(p.layout_manager)
            r = StreamRenderer(
                p.console,
                layout_manager=proxy,
                emit_hook=lambda c: p.tasks.append_task_main_content(tid, c),
                save_log=False,
            )
            r.start(str(agent or ""), str(model or ""), str(language or ""), p.op_name)
            r.add_chunk(str(response or ""))
            r.finish()
        except Exception as e:
            # 缓存失败不应影响主流程；此时至少保证还有 response_done 行
            textual_log.debug(
                "[Presenter] cache_llm_response_for_replay failed; ignore",
                task_id=tid,
                exc_info=e,
            )

    def _debug_stream_route(self, **fields: Any) -> None:
        if str(os.getenv("AIKG_DEBUG_STREAM_ROUTE", "") or "").strip() not in [
            "1",
            "true",
            "yes",
            "y",
        ]:
            return
        try:
            # 用 INFO，避免默认日志级别过滤掉调试信息（尤其在 TUI/devtools 环境下）
            logger.info(
                "[Presenter][StreamRoute] %s",
                " ".join(f"{k}={fields.get(k)!r}" for k in sorted(fields)),
            )
            textual_log.info("[StreamRoute] %s", fields)
        except Exception as e:
            textual_log.debug(
                "[Presenter] debug_stream_route failed; ignore", exc_info=e
            )

    def _is_replaying(self) -> bool:
        p = self._p
        return bool(getattr(p, "tasks", None) and p.tasks.replaying)

    def _task_id(self, message: Any, *, hint_attr: str) -> Optional[str]:
        p = self._p
        hint = getattr(message, hint_attr, "") or ""
        return p.tasks.task_id_from_message(message, node_hint=hint)

    def _capture_task_label(self, task_id: Optional[str], message: Any) -> None:
        """记录 server 下发的 task label（client 只展示）。"""
        p = self._p
        tid = str(task_id or "").strip()
        if not tid:
            return
        label = str(getattr(message, "task_label", "") or "").strip()
        if not label:
            meta = getattr(message, "metadata", None)
            if isinstance(meta, dict):
                label = str(
                    meta.get("task_label")
                    or meta.get("label")
                    or meta.get("task_label_name")
                    or ""
                ).strip()
        if label:
            try:
                p.tasks.set_task_label(tid, label)
            except Exception as e:
                textual_log.debug(
                    "[Presenter] set_task_label failed", task_id=tid, exc_info=e
                )

    def _maybe_emit_main_tab_hint(self, task_id: Optional[str]) -> None:
        """在 main tab 提示其他任务已输出（避免刷屏）。"""
        p = self._p
        tid = str(task_id or "").strip()
        if not tid or tid in ["main", "task_init"]:
            return
        if tid in self._main_tab_hint_tasks:
            return
        watch = str(getattr(p.tasks, "watch_task_id", "") or "").strip()
        if watch not in ["", "main"]:
            return
        label = str(p.tasks.task_label(tid) or "").strip()
        if not label:
            return
        self._main_tab_hint_tasks.add(tid)
        hint = t("presenter.hint.other_tab_output", label=label)
        self._emit_main("main", Text.from_markup(hint))

    def _resolve_llm_stream_task_id(self, message: LLMStreamMessage) -> Optional[str]:
        """
        llm_stream 可能不带 task_id（可能会被 TaskIdPolicy 归到 watch_task_id），
        这里优先通过“最近一次 llm_start 的 agent->task 映射”或“正在 running 的 llm_state”反推归属，
        避免切 tab 后后台流被错归到前台。
        """
        raw = str(getattr(message, "task_label", "") or "").strip()
        if raw:
            return self._task_id(message, hint_attr="agent")

        agent = str(getattr(message, "agent", "") or "").strip()
        if not agent:
            return self._task_id(message, hint_attr="agent")

        mapped = str(self._llm_agent_task.get(agent, "") or "").strip()
        if mapped:
            return mapped

        candidates: list[str] = []
        try:
            for tid in self._p.tasks.sorted_task_ids():
                st = self._p.tasks.task_llm_state(str(tid or ""))
                if not isinstance(st, dict):
                    continue
                if bool(st.get("running")) and str(st.get("agent") or "") == agent:
                    candidates.append(str(tid))
        except Exception as e:
            textual_log.debug(
                "[Presenter] resolve llm_stream task_id failed; fallback", exc_info=e
            )
            candidates = []

        if len(candidates) == 1:
            return candidates[0]
        # 多个候选时：降级为原策略（通常是当前 watch）
        return self._task_id(message, hint_attr="agent")

    def _resolve_llm_task_id(
        self, message: Any, *, agent_attr: str = "agent"
    ) -> Optional[str]:
        """
        给 llm_start/llm_end 使用的 task_id 归属解析：
        - 若 message 带 task_id：沿用 TaskIdPolicy
        - 若不带：优先用 on_llm_start 记录的 agent->task 映射（避免切 tab 后错归）
        """
        raw = str(getattr(message, "task_label", "") or "").strip()
        if raw:
            return self._task_id(message, hint_attr=agent_attr)
        agent = str(getattr(message, agent_attr, "") or "").strip()
        mapped = str(self._llm_agent_task.get(agent, "") or "").strip()
        if mapped:
            return mapped
        return self._task_id(message, hint_attr=agent_attr)

    def _should_render_task(self, task_id: Optional[str]) -> bool:
        return self._p.tasks.should_render_task(task_id)

    def _commit_main(self, task_id: Optional[str], content: Any) -> None:
        """
        最终写入主内容：更新缓存并按 watch/render policy 渲染到当前 UI。
        由 UI intent 消费端调用，避免重复入队。
        """
        p = self._p
        tid = str(task_id or "").strip()
        if not tid:
            tid = str(getattr(p.tasks, "watch_task_id", "") or "").strip() or "main"
        try:
            p.tasks.ensure_task_known(tid)
        except Exception as e:
            textual_log.warning(
                "[Presenter] ensure_task_known failed", task_id=tid, exc_info=e
            )
        try:
            p.tasks.append_task_main_content(tid, content)
        except Exception as e:
            textual_log.warning(
                "[Presenter] append_task_main_content failed",
                task_id=tid,
                exc_info=e,
            )
        if self._should_render_task(tid):
            try:
                p.layout_manager.update_main_content(content)
            except Exception as e:
                textual_log.warning(
                    "[Presenter] update_main_content failed", exc_info=e
                )

    def _enqueue_main(self, task_id: Optional[str], content: Any) -> None:
        """统一入队主内容（优先走 UI intent 队列）。"""
        p = self._p
        mgr = getattr(p, "layout_manager", None) or getattr(p, "textual_manager", None)
        app = getattr(mgr, "app", None) if mgr is not None else None
        q = getattr(app, "ui_event_queue", None) if app is not None else None
        if q is not None:
            try:
                from ..ui.intents import WriteMainContent

                q.put_nowait(WriteMainContent(content=content, task_id=task_id))
                return
            except Exception as e:
                textual_log.debug(
                    "[Presenter] enqueue WriteMainContent failed; drop",
                    exc_info=e,
                )

    def _render_main_direct(self, content: Any) -> None:
        """直接渲染主内容（不入队、不写缓存），用于回放等内部场景。"""
        p = self._p
        try:
            p.layout_manager.update_main_content(content)
        except Exception as e:
            textual_log.warning("[Presenter] render_main_direct failed", exc_info=e)

    def _emit_main(self, task_id: Optional[str], content: Any) -> None:
        """
        统一出口：主内容统一入队，由 UI intent 消费端完成缓存与渲染。
        """
        self._enqueue_main(task_id, content)

    def _emit_main_global(self, content: Any) -> None:
        """
        没有明确 task_id 的“全局输出”也要进入可回放链路。
        约定：优先归到当前 watch_task_id，否则落到 `_default_` 桶。
        """
        self._emit_main(None, content)

    def _record_llm_start(
        self, task_id: Optional[str], message: LLMStartMessage
    ) -> None:
        p = self._p
        if not task_id:
            return
        p.tasks.ensure_task_known(task_id)
        if not p.tasks.watch_task_id and task_id not in ["task_init", "main"]:
            # evolve 并发下：首次出现有效 task_id 时自动设为默认 watch，避免多路输出交叉
            p.tasks.watch_task_id = task_id
        p.tasks.record_task_event(
            task_id,
            {"type": "llm_start", "agent": message.agent, "model": message.model},
        )
        st = p.tasks.task_llm_state(task_id) or {}
        st.update(
            {
                "running": True,
                "agent": message.agent,
                "model": message.model,
                "language": _llm_language_from_agent(message.agent),
            }
        )

    def _record_llm_end(self, task_id: Optional[str], message: LLMEndMessage) -> None:
        p = self._p
        if not task_id:
            return
        p.tasks.ensure_task_known(task_id)
        lang = _llm_language_from_agent(message.agent)
        p.tasks.record_task_event(
            task_id,
            {
                "type": "llm_end",
                "agent": message.agent,
                "model": message.model,
                "duration": message.duration,
                "prompt_tokens": message.prompt_tokens,
                "reasoning_tokens": message.reasoning_tokens,
                "output_tokens": message.output_tokens,
                "total_tokens": message.total_tokens,
                "language": lang,
                "response": message.response,
            },
        )
        st = p.tasks.task_llm_state(task_id) or {}
        st.update(
            {
                "running": False,
            }
        )

    # 约束：不要在 handlers 内直接调用 p._update_layout 写主输出，
    # 所有“可见输出”必须走 _emit_main(...) 以保证切 tab 回放一致。

    def _render_node_end_coder(
        self, task_id: Optional[str], message: NodeEndMessage
    ) -> None:
        p = self._p
        output_text = (
            f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
            f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}] - {t('presenter.code_generated')}"
        )
        self._emit_main(task_id, Text.from_markup(output_text))

        kernel_code = None
        if isinstance(message.result, dict):
            kernel_code = message.result.get("coder_code", "")

    def _render_node_end_designer(
        self, task_id: Optional[str], message: NodeEndMessage
    ) -> None:
        p = self._p
        output_text = (
            f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
            f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}] - {t('presenter.sketch_generated')}"
        )
        self._emit_main(task_id, Text.from_markup(output_text))

        sketch = None
        if isinstance(message.result, dict):
            sketch = message.result.get("designer_code", "") or message.result.get(
                "sketch", ""
            )

    def _render_node_end_conductor(
        self, task_id: Optional[str], message: NodeEndMessage
    ) -> None:
        p = self._p
        output_text = (
            f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
            f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}] - {t('presenter.error_analysis_done')}"
        )
        self._emit_main(task_id, Text.from_markup(output_text))

    def _render_node_end_verifier(
        self, task_id: Optional[str], message: NodeEndMessage
    ) -> None:
        p = self._p
        verify_result: Optional[bool] = None
        verifier_error = ""
        if isinstance(message.result, dict):
            verify_result = _normalize_boolish(
                message.result.get("verifier_result", None)
            )
            verifier_error = str(message.result.get("verifier_error") or "")

        if verify_result is True:
            output_text = (
                f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
                f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}] - {t('presenter.verify_pass')}"
            )
            self._emit_main(task_id, Text.from_markup(output_text))

            if not self._is_replaying() and isinstance(message.result, dict):
                profile_res = message.result.get("profile_res", {})
                if profile_res and isinstance(profile_res, dict):
                    gen_time = profile_res.get("gen_time", 0.0)
                    base_time = profile_res.get("base_time", 0.0)
                    speedup = profile_res.get("speedup", 0.0)

                    if gen_time > 0 or base_time > 0 or speedup > 0:
                        p.performance_history.append(
                            {
                                "round": len(p.performance_history) + 1,
                                "gen_time": gen_time,
                                "base_time": base_time,
                                "speedup": speedup,
                            }
                        )

                    if gen_time > 0 or base_time > 0 or speedup > 0:
                        speedup_color = (
                            DisplayStyle.BOLD_GREEN
                            if speedup > 1.0
                            else DisplayStyle.YELLOW
                        )
                        perf_info: list[str] = []
                        if base_time > 0:
                            perf_info.append(
                                f"{t('presenter.perf.baseline')}: {base_time:.2f} µs"
                            )
                        if gen_time > 0:
                            perf_info.append(
                                f"{t('presenter.perf.optimized')}: {gen_time:.2f} µs"
                            )
                        elif gen_time == 0.0 and speedup > 0:
                            perf_info.append(
                                f"{t('presenter.perf.optimized')}: {gen_time:.2f} µs"
                            )
                        if speedup > 0:
                            perf_info.append(
                                f"{t('presenter.perf.speedup')}: [{speedup_color}]{speedup:.2f}x[/{speedup_color}]"
                            )
                        if perf_info:
                            perf_output = f"  {UISymbol.TREE_END} {t('presenter.perf.performance')}: {' | '.join(perf_info)}"
                            self._emit_main(task_id, Text.from_markup(perf_output))

        elif verify_result is False or (
            verify_result is None and verifier_error.strip()
        ):
            output_text = (
                f"[{DisplayStyle.YELLOW}]{message.node} {t('presenter.done')}[/{DisplayStyle.YELLOW}] - {t('presenter.duration')}: "
                f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}] - {t('presenter.verify_fail')}"
            )
            self._emit_main(task_id, Text.from_markup(output_text))
            if verifier_error.strip():
                self._emit_main(
                    task_id,
                    Text.from_markup(
                        f"[{DisplayStyle.BOLD_RED}]{t('presenter.verifier_error_log')}[/{DisplayStyle.BOLD_RED}]"
                    ),
                )
                self._emit_main(
                    task_id,
                    SyntaxBlockMainContent(
                        code=verifier_error, lexer_name="pytb", add_end_separator=True
                    ),
                )
        else:
            output_text = (
                f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
                f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}]"
            )
            self._emit_main(task_id, Text.from_markup(output_text))

    def on_llm_start(self, message: LLMStartMessage) -> None:
        p = self._p
        replaying = self._is_replaying()
        task_id = self._resolve_llm_task_id(message, agent_attr="agent")
        self._capture_task_label(task_id, message)
        if task_id and (not replaying):
            self._record_llm_start(task_id, message)

        # 即使不是当前 watch task，也要把输出缓存到对应 task，确保多 tab 实时更新
        p.current_model = message.model
        p.llm_running = True
        p.last_reasoning_tokens = None
        p.last_output_tokens = None
        p.last_total_tokens = None
        p.last_prompt_tokens = None
        output = f"  [{DisplayStyle.DIM}]├─ {t('presenter.call_llm')}: {message.model}[/{DisplayStyle.DIM}]"
        self._emit_main(task_id, Text.from_markup(output))
        if task_id:
            self._llm_agent_task[str(message.agent or "")] = str(task_id)
        if p.use_stream and task_id:
            active = bool(self._should_render_task(task_id)) and (not replaying)
            if active:
                p.llm_buffer = ""
            try:
                p.tasks.stream_session(str(task_id or "")).on_llm_start(
                    agent=str(message.agent or ""),
                    model=str(message.model or ""),
                    language=_llm_language_from_agent(str(message.agent or "")),
                    active=bool(active),
                )
            except Exception as e:
                textual_log.warning(
                    "[Presenter] stream_session.on_llm_start failed",
                    task_id=str(task_id or ""),
                    exc_info=e,
                )

    def on_llm_end(self, message: LLMEndMessage) -> None:
        p = self._p
        replaying = self._is_replaying()
        task_id = self._resolve_llm_task_id(message, agent_attr="agent")
        self._capture_task_label(task_id, message)
        if task_id and (not replaying):
            self._record_llm_end(task_id, message)

        if not replaying:
            p.llm_running = False
            p.last_prompt_tokens = message.prompt_tokens
            p.last_reasoning_tokens = message.reasoning_tokens
            p.last_output_tokens = message.output_tokens
            p.last_total_tokens = message.total_tokens
            p.llm_records.append(
                {
                    "agent": getattr(message, "agent", ""),
                    "prompt_tokens": message.prompt_tokens,
                    "reasoning_tokens": message.reasoning_tokens,
                    "output_tokens": message.output_tokens,
                    "total_tokens": message.total_tokens,
                    "duration": message.duration,
                    "raw_total_tokens": message.total_tokens,
                }
            )

        if p.use_stream:
            active = bool(self._should_render_task(task_id))
            # 非 active：把完整 response 离屏渲染进缓存，保证切回可见
            if (not replaying) and (not active) and message.response:
                self._cache_llm_response_for_replay(
                    task_id,
                    agent=str(getattr(message, "agent", "") or ""),
                    model=str(getattr(message, "model", "") or ""),
                    language=_llm_language_from_agent(
                        str(getattr(message, "agent", "") or "")
                    ),
                    response=str(getattr(message, "response", "") or ""),
                )
            if task_id:
                try:
                    p.tasks.stream_session(str(task_id or "")).on_llm_end(
                        active=bool(active),
                        replaying=bool(replaying),
                        response=str(getattr(message, "response", "") or ""),
                    )
                except Exception as e:
                    textual_log.warning(
                        "[Presenter] stream_session.on_llm_end failed",
                        task_id=str(task_id or ""),
                        exc_info=e,
                    )
            if active and (not replaying):
                p.llm_buffer = ""
            ag = str(getattr(message, "agent", "") or "").strip()
            if ag and self._llm_agent_task.get(ag) == str(task_id or ""):
                self._llm_agent_task.pop(ag, None)
            # “response_done” 行属于可见输出：每个 task 都要缓存，保证切换后看到一致的收尾行
            tokens_text = _format_token_info(
                prompt_tokens=message.prompt_tokens,
                reasoning_tokens=message.reasoning_tokens,
                output_tokens=message.output_tokens,
                total_tokens=message.total_tokens,
            )
            output = f"  [{DisplayStyle.DIM}]└─ {t('presenter.response_done')}{tokens_text}[/{DisplayStyle.DIM}]\n"
            self._emit_main(task_id, Text.from_markup(output))
        else:
            preview = (
                message.response[:100] + "..."
                if len(message.response) > 100
                else message.response
            )
            tokens_text = _format_token_info(
                prompt_tokens=message.prompt_tokens,
                reasoning_tokens=message.reasoning_tokens,
                output_tokens=message.output_tokens,
                total_tokens=message.total_tokens,
            )
            output = f"  [{DisplayStyle.DIM}]└─ {t('presenter.response')} ({message.duration:.2f}s): {preview}{tokens_text}[/{DisplayStyle.DIM}]"
            self._emit_main(task_id, Text.from_markup(output))

    def on_llm_stream(self, message: LLMStreamMessage) -> None:
        p = self._p
        task_id = self._resolve_llm_stream_task_id(message)
        self._capture_task_label(task_id, message)
        self._debug_stream_route(
            phase="llm_stream",
            raw_tid=str(getattr(message, "task_label", "") or ""),
            resolved_tid=str(task_id or ""),
            agent=str(getattr(message, "agent", "") or ""),
            watch=str(p.tasks.watch_task_id or ""),
            active=bool(self._should_render_task(task_id)),
        )
        if p.use_stream and task_id:
            active = bool(self._should_render_task(task_id))
            try:
                p.tasks.stream_session(str(task_id or "")).on_llm_stream(
                    str(getattr(message, "chunk", "") or ""),
                    is_reasoning=bool(getattr(message, "is_reasoning", False)),
                    active=bool(active),
                )
            except Exception as e:
                textual_log.warning(
                    "[Presenter] stream_session.on_llm_stream failed",
                    task_id=str(task_id or ""),
                    exc_info=e,
                )
            if active:
                p.llm_buffer += str(getattr(message, "chunk", "") or "")

    def on_job_submitted(self, job_id: str) -> None:
        p = self._p
        if not job_id:
            return
        p.last_job_id = job_id
        output = f"[{DisplayStyle.DIM}]job_id: {job_id}[/{DisplayStyle.DIM}]"
        p._refresh_info_panel()
        p._refresh_workflow_panel()
        self._emit_main_global(Text.from_markup(output))

    def on_progress(self, message: ProgressMessage) -> None:
        p = self._p
        p.tasks.poll_ui_events()
        data0 = getattr(message, "data", {}) or {}
        logger.info(
            "[Presenter] on_progress scope=%s round=%s done=%s total=%s",
            str(getattr(message, "scope", "") or ""),
            str(getattr(data0, "get", lambda *_: "")("round", "")),
            str(getattr(data0, "get", lambda *_: "")("done", "")),
            str(getattr(data0, "get", lambda *_: "")("total", "")),
        )
        data = getattr(message, "data", {}) or {}
        try:
            round_idx = int(getattr(data, "get", lambda *_: 0)("round", 0) or 0)
        except (TypeError, ValueError) as e:
            textual_log.debug(
                "[Presenter] progress round cast failed; fallback 0", exc_info=e
            )
            round_idx = 0
        try:
            p.tasks.evolve_round_snapshots[round_idx] = dict(data)
        except Exception as e:
            textual_log.warning(
                "[Presenter] store evolve_round_snapshots failed", exc_info=e
            )
        # 进度里也可能包含任务列表：提前纳入可切换列表
        tmap = (
            data.get("tasks")
            if isinstance(data, dict) and isinstance(data.get("tasks"), dict)
            else {}
        )
        try:
            for tid in tmap.keys():
                p.tasks.ensure_task_known(str(tid))
        except Exception as e:
            textual_log.debug(
                "[Presenter] ensure_task_known from progress failed", exc_info=e
            )

        try:
            p.tasks.latest_progress_data = dict(data)
        except Exception as e:
            textual_log.warning(
                "[Presenter] set latest_progress_data failed", exc_info=e
            )
            p.tasks.latest_progress_data = {}

        # 进度是“全局聚合信息”，不应依赖 watch/回放状态才能刷新；
        # 这里直接刷新右侧面板，避免 refresh_progress_panel 的条件导致断更。
        try:
            p._refresh_info_panel()
        except Exception as e:
            textual_log.warning("[Presenter] _refresh_info_panel failed", exc_info=e)
        try:
            p._refresh_workflow_panel()
        except Exception as e:
            textual_log.warning(
                "[Presenter] _refresh_workflow_panel failed", exc_info=e
            )
        try:
            p.tasks.refresh_progress_panel()
        except Exception as e:
            textual_log.warning("[Presenter] refresh_progress_panel failed", exc_info=e)

    def on_node_start(self, message: NodeStartMessage) -> None:
        p = self._p
        replaying = self._is_replaying()
        p.tasks.poll_ui_events()
        task_id = self._task_id(message, hint_attr="node")
        self._capture_task_label(task_id, message)
        if task_id and (not replaying):
            p.tasks.ensure_task_known(task_id)
            event_idx = p.tasks.record_task_event(
                task_id, {"type": "node_start", "node": message.node}
            )
            # 更新该 task 的节点态（即使当前不渲染，也要保持可切换回放）
            try:
                st = p.tasks.ensure_task_node_state(task_id)
                node = str(message.node or "")
                if node:
                    st["current_node"] = node
                    st["status"] = "running"

                    rc = st.get("run_counts")
                    if not isinstance(rc, dict):
                        rc = {}
                    try:
                        rc[node] = int(rc.get(node, 0) or 0) + 1
                    except (TypeError, ValueError) as e:
                        textual_log.debug(
                            "[Presenter] run_counts cast failed; fallback +1",
                            exc_info=e,
                        )
                        rc[node] = 1
                    st["run_counts"] = rc
                    st["current_run_no"] = int(rc.get(node, 0) or 0)

                    seen = st.get("seen_nodes")
                    if not isinstance(seen, list):
                        seen = []
                    if node not in seen:
                        seen.append(node)
                    st["seen_nodes"] = seen
            except Exception as e:
                textual_log.warning(
                    "[Presenter] update task node state failed",
                    task_id=str(task_id or ""),
                    exc_info=e,
                )

            # 全局 Trace：仅记录 node_start（跨 task）
            try:
                st = p.tasks.ensure_task_node_state(task_id)
                node = str(message.node or "")
                rn = int(st.get("current_run_no") or 0)
                # 在 chat 输出队列里插入锚点标记（用于“滚动跳转”）
                try:
                    self._emit_main(
                        task_id,
                        TraceAnchorMainContent(
                            task_id=str(task_id or ""), event_idx=int(event_idx or 0)
                        ),
                    )
                except Exception as e:
                    textual_log.warning(
                        "[Presenter] emit trace anchor failed", exc_info=e
                    )

                p.tasks.record_trace_node_start(
                    task_id=task_id, node=node, run_no=rn, event_idx=int(event_idx or 0)
                )
            except Exception as e:
                textual_log.warning(
                    "[Presenter] record_trace_node_start failed", exc_info=e
                )

            # 初始化默认观察目标（第一次遇到匹配的 task_id）
            if not p.tasks.watch_task_id and task_id not in ["task_init"]:
                p.tasks.watch_task_id = task_id
                p.tasks.refresh_task_tabs()
                p.tasks.refresh_trace_panel()

            self._maybe_emit_main_tab_hint(task_id)

        if not replaying:
            p.current_agent = message.node

            # 同步当前 task 的节点态到 UI
            if task_id and self._should_render_task(task_id):
                p.tasks.apply_task_node_state(task_id)
        else:
            # 回放：允许从 message.state 注入 event_idx（用于 trace_anchor）
            try:
                st = getattr(message, "state", None)
                if isinstance(st, dict) and "_event_idx" in st:
                    self._emit_main(
                        task_id,
                        TraceAnchorMainContent(
                            task_id=str(task_id or ""),
                            event_idx=int(st.get("_event_idx") or 0),
                        ),
                    )
            except Exception as e:
                textual_log.debug(
                    "[Presenter] replay trace anchor injection failed", exc_info=e
                )

        output = f"\n[{DisplayStyle.BOLD_CYAN}]{t('presenter.node')}: {message.node}[/{DisplayStyle.BOLD_CYAN}]"
        self._emit_main(task_id, Text.from_markup(output))
        if (not replaying) and task_id and self._should_render_task(task_id):
            p._refresh_info_panel()
            p._refresh_workflow_panel()

    def on_node_end(self, message: NodeEndMessage) -> None:
        p = self._p
        replaying = self._is_replaying()
        p.tasks.poll_ui_events()
        task_id = self._task_id(message, hint_attr="node")
        self._capture_task_label(task_id, message)
        if task_id and (not replaying):
            p.tasks.ensure_task_known(task_id)
            node_end_payload = {
                "type": "node_end",
                "node": message.node,
                "duration": message.duration,
            }
            if message.node == NodeName.VERIFIER and isinstance(message.result, dict):
                vr_norm = _normalize_boolish(
                    message.result.get("verifier_result", None)
                )
                node_end_payload.update(
                    {
                        "verifier_result": vr_norm,
                        "verifier_error": str(
                            message.result.get("verifier_error") or ""
                        ),
                    }
                )
            p.tasks.record_task_event(task_id, node_end_payload)
            # 更新该 task 的节点态
            try:
                st = p.tasks.ensure_task_node_state(task_id)
                node = str(message.node or "")
                if node:
                    # 如果结束的是当前节点，标记 done；否则不强行覆盖 current_node
                    if str(st.get("current_node") or "") == node:
                        st["status"] = "done"
                    # seen_nodes 补齐（防止只收到 end）
                    seen = st.get("seen_nodes")
                    if not isinstance(seen, list):
                        seen = []
                    if node not in seen:
                        seen.append(node)
                    st["seen_nodes"] = seen
            except Exception as e:
                textual_log.warning(
                    "[Presenter] update node_end state failed",
                    task_id=str(task_id or ""),
                    exc_info=e,
                )
            # 如果是 verifier，记录最终状态（用于 summary 展示）
            if message.node == NodeName.VERIFIER and isinstance(message.result, dict):
                vr = _normalize_boolish(message.result.get("verifier_result", None))
                ve = message.result.get("verifier_error") or ""
                p.tasks.evolve_task_summary.setdefault(task_id, {})
                p.tasks.evolve_task_summary[task_id].update(
                    {
                        "task_id": task_id,
                        "verifier_result": vr,
                        "verifier_error": str(ve),
                    }
                )

        if not replaying:
            if task_id and self._should_render_task(task_id):
                p.tasks.apply_task_node_state(task_id)

        # 收集节点耗时（合并了原 agent 的耗时统计）
        if not p.tasks.replaying:
            p.node_timings.append(
                {"node": message.node, "duration": round(message.duration, 2)}
            )

        if message.node == NodeName.CODER:
            if task_id and self._should_render_task(task_id):
                self._render_node_end_coder(task_id, message)
            else:
                output_text = (
                    f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
                    f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}] - {t('presenter.code_generated')}"
                )
                self._emit_main(task_id, Text.from_markup(output_text))
        elif message.node == NodeName.VERIFIER:
            # 注意：无论当前是否在 watch 该 task，都应生成一致的 main_content 序列，
            # 以保证切换 tab 回放时（replay_task_from_cache）不会丢失 verifier 的性能行等输出。
            # _emit_main(...) 会先写入 task 缓存，再按 should_render_task 决定是否渲染到当前 UI。
            if task_id:
                self._render_node_end_verifier(task_id, message)
            else:
                self._render_node_end_verifier(None, message)
        elif message.node == NodeName.DESIGNER:
            if task_id and self._should_render_task(task_id):
                self._render_node_end_designer(task_id, message)
            else:
                output_text = (
                    f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
                    f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}] - {t('presenter.sketch_generated')}"
                )
                self._emit_main(task_id, Text.from_markup(output_text))
        elif message.node == NodeName.CONDUCTOR:
            if task_id and self._should_render_task(task_id):
                self._render_node_end_conductor(task_id, message)
            else:
                output_text = (
                    f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
                    f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}] - {t('presenter.error_analysis_done')}"
                )
                self._emit_main(task_id, Text.from_markup(output_text))
        else:
            output_text = (
                f"[{DisplayStyle.GREEN}]{message.node} {t('presenter.done')}[/{DisplayStyle.GREEN}] - {t('presenter.duration')}: "
                f"[{DisplayStyle.YELLOW}]{message.duration:.2f}s[/{DisplayStyle.YELLOW}]"
            )
            self._emit_main(task_id, Text.from_markup(output_text))

        if (not replaying) and task_id and self._should_render_task(task_id):
            p._refresh_info_panel()
            p._refresh_workflow_panel()

    def display_summary(self, result: dict) -> None:
        p = self._p
        from .summary import SummaryRenderer

        p.tasks.inject_evolve_metadata(result)

        width = max(80, (p.console.width or 120))
        tmp_console = RichConsole(
            record=True, width=width, color_system=None, force_terminal=False
        )
        SummaryRenderer(tmp_console).display(
            result=result,
            llm_records=p.llm_records or [],
            node_timings=p.node_timings or [],
            performance_history=p.performance_history or [],
        )
        rendered = tmp_console.export_text(clear=False)
        self._emit_main_global(Text(rendered))
