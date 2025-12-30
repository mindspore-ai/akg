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

from textual import log
from rich.text import Text

from ...ui.intents import (
    AppMounted,
    CancelRequest,
    LangChanged,
    ThemeChanged,
    TraceJump,
    WatchNext,
    WatchSet,
    WriteMainContent,
)
from ...constants import make_gradient_logo
from ...utils.i18n import set_lang


class UiIntentController:
    """把 UI intent（快捷键/点击）转成对 Watch/Trace 的调用。"""

    def __init__(
        self,
        presenter,
        store,
        *,
        watch_next,
        set_watch_task,
        refresh_progress_panel,
        refresh_trace_panel,
    ) -> None:
        self._p = presenter
        self._s = store
        self._watch_next = watch_next
        self._set_watch_task = set_watch_task
        self._refresh_progress_panel = refresh_progress_panel
        self._refresh_trace_panel = refresh_trace_panel

    def start_ui_pump(self) -> None:
        """启动 UI 事件轮询任务（Textual 运行时）。"""
        if self._s.ui_pump_running:
            return
        self._s.ui_pump_running = True
        try:
            import asyncio

            async def _pump():
                while self._s.ui_pump_running:
                    try:
                        self.poll_ui_events()
                    except Exception as e:
                        log.debug(
                            "[UIIntent] poll_ui_events failed; continue", exc_info=e
                        )
                    await asyncio.sleep(0.1)

            self._s.ui_pump_task = asyncio.create_task(_pump())
        except Exception as e:
            log.warning("[UIIntent] start_ui_pump failed", exc_info=e)
            self._s.ui_pump_task = None
            self._s.ui_pump_running = False

    def stop_ui_pump(self) -> None:
        """停止 UI 事件轮询任务。"""
        self._s.ui_pump_running = False
        try:
            t = self._s.ui_pump_task
            if t is not None and hasattr(t, "cancel"):
                t.cancel()
        except Exception as e:
            log.debug("[UIIntent] stop_ui_pump cancel failed", exc_info=e)
        self._s.ui_pump_task = None

    # ======= UI events =======

    def poll_ui_events(self) -> None:
        """轮询 UI 快捷键事件（类型化 intent）。"""
        p = self._p
        mgr = getattr(p, "layout_manager", None) or getattr(p, "textual_manager", None)
        if mgr is None or not hasattr(mgr, "drain_ui_intents"):
            return

        try:
            intents = mgr.drain_ui_intents()
        except Exception as e:
            log.debug("[UIIntent] drain_ui_intents failed; ignore", exc_info=e)
            return

        for intent in intents:
            if isinstance(intent, AppMounted):
                try:
                    app = getattr(getattr(p, "layout_manager", None), "app", None)
                    if app is not None and getattr(app, "logo_printed", False):
                        continue
                    p._handlers._emit_main_global(make_gradient_logo())
                    p._handlers._emit_main_global(Text("\n"))
                    if app is not None:
                        app.logo_printed = True
                except Exception as e:
                    log.debug("[UIIntent] handle AppMounted failed", exc_info=e)
                continue
            if isinstance(intent, WatchNext):
                self._watch_next()
                continue
            if isinstance(intent, WatchSet):
                tid = str(getattr(intent, "task_id", "") or "").strip()
                if tid:
                    self._set_watch_task(tid)
                continue
            if isinstance(intent, TraceJump):
                tid = str(getattr(intent, "task_id", "") or "").strip()
                eidx = getattr(intent, "event_idx", None)
                eidx_i = None
                try:
                    eidx_i = int(eidx) if eidx is not None else None
                except (TypeError, ValueError) as e:
                    log.debug("[UIIntent] event_idx cast failed; ignore", exc_info=e)
                    eidx_i = None
                if tid:
                    self._set_watch_task(tid, force=True, upto_event_idx=eidx_i)
                continue
            if isinstance(intent, LangChanged):
                lang = str(getattr(intent, "lang", "") or "").strip()
                if lang:
                    try:
                        set_lang(lang)
                    except Exception as e:
                        log.warning(
                            "[UIIntent] set_lang failed; ignore", lang=lang, exc_info=e
                        )
                # best-effort 刷新右侧面板/Trace 标题/结构化进度
                try:
                    self._refresh_progress_panel()
                except Exception as e:
                    log.debug(
                        "[UIIntent] refresh_progress_panel failed; fallback refresh panels",
                        exc_info=e,
                    )
                    try:
                        p._refresh_info_panel()
                        p._refresh_workflow_panel()
                    except Exception as e2:
                        log.debug(
                            "[UIIntent] fallback refresh panels failed", exc_info=e2
                        )
                try:
                    self._refresh_trace_panel()
                except Exception as e:
                    log.debug("[UIIntent] refresh_trace_panel failed", exc_info=e)
                continue
            if isinstance(intent, CancelRequest):
                try:
                    p.request_cancel(
                        reason=str(getattr(intent, "reason", "") or "cancelled by ctrl+c")
                    )
                except Exception as e:
                    log.debug("[UIIntent] cancel request failed", exc_info=e)
                continue
            if isinstance(intent, ThemeChanged):
                tid = str(getattr(self._p.tasks, "watch_task_id", "") or "") or "main"
                try:
                    self._set_watch_task(tid, force=True)
                except Exception as e:
                    log.debug("[UIIntent] handle ThemeChanged failed", exc_info=e)
                continue
            if isinstance(intent, WriteMainContent):
                content = getattr(intent, "content", None)
                if content is not None:
                    try:
                        task_id = getattr(intent, "task_id", None)
                        p._handlers._commit_main(task_id, content)
                    except Exception as e:
                        log.debug("[UIIntent] handle WriteMainContent failed", exc_info=e)
                continue
