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

from typing import Any

from textual import log

from ai_kernel_generator.cli.cli.presenter.stream import StreamRenderer


class TaskStreamSession:
    """每个 task 一份的流式渲染会话（renderer + started + buffer）。

    目标：把“是否 start / 何时 flush / 用哪个 renderer”这类易踩坑的状态收拢到一个对象里，
    上层（handlers/watch）只传入事件与 active 状态，不再直接读写零散字段。
    """

    def __init__(self, presenter: Any, task_id: str, *, append_main_content) -> None:
        self._p = presenter
        self._task_id = str(task_id or "").strip()
        self._append_main_content = append_main_content

        self._renderer: StreamRenderer | None = None
        self._started: bool = False
        self._buffer_chunks: list[str] = []

        # 记录最近一次 llm_start 的元信息，供“切回后补渲染/首 chunk start”使用
        self._agent: str = ""
        self._model: str = ""
        self._language: str = ""

    @property
    def task_id(self) -> str:
        return self._task_id

    def buffer_debug_snapshot(self, *, max_len: int = 160) -> dict[str, Any]:
        """返回当前 buffer 的可诊断信息（用于日志，不改变状态）。"""
        try:
            chunk_count = len(self._buffer_chunks)
            buf = "".join(self._buffer_chunks) if chunk_count else ""
            stripped = buf.lstrip()

            def _preview(s: str) -> str:
                v = str(s or "")
                v = v.replace("\n", "\\n").replace("\r", "\\r").replace("\t", "\\t")
                if len(v) > max_len:
                    return v[:max_len] + "…"
                return v

            return {
                "chunks": chunk_count,
                "buf_len": len(buf),
                "has_literal_newline": ("\\n" in buf),
                "has_real_newline": ("\n" in buf),
                "starts_like_json": stripped.startswith("{")
                or stripped.startswith("```json"),
                "has_json_code_key": (
                    '"code": "' in buf
                    or '"task_code": "' in buf
                    or '"sketch": "' in buf
                ),
                "head": _preview(stripped[:200]),
                "tail": _preview(stripped[-200:]),
            }
        except Exception as e:
            log.debug(
                "[TaskStreamSession] buffer_debug_snapshot failed; fallback empty",
                task_id=self._task_id,
                exc_info=e,
            )
            return {"chunks": 0, "buf_len": 0}

    def on_llm_start(
        self, *, agent: str, model: str, language: str, active: bool
    ) -> None:
        self._agent = str(agent or "")
        self._model = str(model or "")
        self._language = str(language or "")
        self._started = False
        self._buffer_chunks.clear()

        if not active:
            return

        r = self._ensure_renderer()
        r.start(self._agent, self._model, self._language, str(self._p.op_name or ""))
        self._started = True

    def on_llm_stream(self, chunk: str, *, active: bool) -> None:
        c = str(chunk or "")
        if not c:
            return
        if not active:
            self._buffer_chunks.append(c)
            return

        # active：先把之前积累的 buffer 补上（避免切 tab 的竞态导致乱序）
        self.flush_buffer(active=True)

        r = self._ensure_renderer()
        if not self._started:
            r.start(
                self._agent, self._model, self._language, str(self._p.op_name or "")
            )
            self._started = True
        r.add_chunk(c)

    def flush_buffer(self, *, active: bool) -> None:
        if not active:
            return
        if not self._buffer_chunks:
            return

        buf = "".join(self._buffer_chunks)
        self._buffer_chunks.clear()
        if not buf:
            return

        r = self._ensure_renderer()
        if not self._started:
            r.start(
                self._agent, self._model, self._language, str(self._p.op_name or "")
            )
            self._started = True
        # 切换回该 task 时：尽快把 buffer 落到 UI（避免刚好撞上 update_interval 看不到）
        try:
            r.last_update_time = 0
        except Exception as e:
            log.debug(
                "[TaskStreamSession] reset last_update_time failed; ignore",
                task_id=self._task_id,
                exc_info=e,
            )
        r.add_chunk(buf)

    def on_llm_end(self, *, active: bool, replaying: bool, response: str) -> None:
        if replaying:
            # replaying：不依赖逐 chunk 回放，直接用 response 一次性渲染
            if active and str(response or "").strip():
                try:
                    r = self._ensure_renderer()
                    r.start(
                        self._agent,
                        self._model,
                        self._language,
                        str(self._p.op_name or ""),
                    )
                    r.add_chunk(str(response or ""))
                    r.finish()
                except Exception as e:
                    log.warning(
                        "[TaskStreamSession] replay render failed",
                        task_id=self._task_id,
                        exc_info=e,
                    )
            self._buffer_chunks.clear()
            self._started = False
            return

        if active and self._renderer is not None:
            try:
                self.flush_buffer(active=True)
                # 正常 active 流式：finish 会 flush 未闭合块并打印完成标记
                self._renderer.finish()
            except Exception as e:
                log.warning(
                    "[TaskStreamSession] finish failed",
                    task_id=self._task_id,
                    exc_info=e,
                )

        # 结束后统一清理（无论 active 与否）
        self._buffer_chunks.clear()
        self._started = False

    def _ensure_renderer(self) -> StreamRenderer:
        if self._renderer is not None:
            return self._renderer

        self._renderer = StreamRenderer(
            self._p.console, layout_manager=self._p.layout_manager
        )
        try:
            self._renderer.set_emit_hook(
                lambda c, _tid=self._task_id: self._append_main_content(_tid, c)
            )
        except Exception as e:
            log.warning(
                "[TaskStreamSession] set_emit_hook failed",
                task_id=self._task_id,
                exc_info=e,
            )
        return self._renderer
