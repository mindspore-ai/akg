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

"""流式渲染器 - 真流式纯文本增量渲染 (重构版)"""

import re
import time
from rich.console import Console
from textual import log

from ai_kernel_generator.cli.cli.constants import Defaults, SyntaxLanguage
from ai_kernel_generator.cli.cli.utils.paths import get_stream_save_dir
from .safety import find_safe_render_position
from .logging import init_stream_renderer_logger
from .painter import ConsolePainter
from .json_processor import JsonStreamProcessor
from .state import RenderState

# ==================================================================================
# 日志配置
# ==================================================================================

logger = init_stream_renderer_logger(__name__)

# ==================================================================================
# 状态管理
# ==================================================================================

# ==================================================================================
# 主渲染器
# ==================================================================================


class StreamRenderer:
    """
    真流式渲染器 - 主入口

    负责协调数据接收、安全切分、解析和绘制。
    """

    def __init__(
        self,
        console: Console,
        layout_manager=None,
        *,
        emit_hook=None,
        save_log: bool = True,
    ):
        self.console = console
        self.layout_manager = layout_manager
        self.painter = ConsolePainter(
            console, layout_manager=layout_manager, emit_hook=emit_hook
        )
        self.json_processor = JsonStreamProcessor(self.painter)
        self._emit_hook = emit_hook
        self.save_log = bool(save_log)
        # 核心数据
        self.full_buffer = ""
        self._pending_chunks: list[tuple[str, bool]] = []
        self._pending_text = ""
        self._pending_dim = False

        # 状态
        self.state = RenderState()
        self.agent_name = ""
        self.op_name = ""
        self.default_lang = SyntaxLanguage.PYTHON

        # 刷新控制
        self.last_update_time = 0
        self.update_interval = Defaults.STREAM_UPDATE_INTERVAL

    def set_emit_hook(self, emit_hook) -> None:
        self._emit_hook = emit_hook
        self.painter._emit_hook = emit_hook

    def start(
        self,
        agent_name: str,
        model_name: str,
        language: str = SyntaxLanguage.PYTHON,
        op_name: str = "",
    ):
        self.agent_name = agent_name
        self.op_name = op_name
        self.default_lang = language
        self.full_buffer = ""
        self._pending_chunks.clear()
        self._pending_text = ""
        self._pending_dim = False
        self.state.reset()

        self.painter.print_divider()

    def add_chunk(self, chunk: str, *, is_reasoning: bool = False):
        c = str(chunk or "")
        if not c:
            return
        self.full_buffer += c
        self._pending_chunks.append((c, bool(is_reasoning)))

        # 限流更新
        now = time.time()
        if now - self.last_update_time > self.update_interval:
            self._render_incremental()
            self.last_update_time = now

    def finish(self):
        # 渲染剩余所有内容
        self._render_incremental(force_all=True)

        # 强制闭合未完成的块
        self._flush_incomplete_blocks()

        self.painter.print_finish_mark(self.agent_name)
        if self.save_log:
            self._save_log()

    def _render_incremental(self, force_all: bool = False):
        """增量渲染循环"""
        if force_all:
            # 强制刷完时合并同类型 chunk，避免 token 级分片导致拆行
            merged: list[tuple[str, bool]] = []
            if self._pending_text:
                merged.append((self._pending_text, bool(self._pending_dim)))
            for text, is_reasoning in self._pending_chunks:
                c = str(text or "")
                if not c:
                    continue
                dim = bool(is_reasoning)
                if merged and merged[-1][1] == dim:
                    merged[-1] = (merged[-1][0] + c, dim)
                else:
                    merged.append((c, dim))

            self._pending_text = ""
            self._pending_dim = False
            self._pending_chunks.clear()

            for text, dim in merged:
                if text:
                    self._process_content_lines(text, dim=dim)
            return

        while True:
            if not self._pending_text:
                if not self._pending_chunks:
                    return
                text, is_reasoning = self._pending_chunks.pop(0)
                self._pending_text = str(text or "")
                self._pending_dim = bool(is_reasoning)
                if not self._pending_text:
                    continue

            # 1. 计算安全渲染位置
            if force_all:
                limit = len(self._pending_text)
            else:
                limit = find_safe_render_position(
                    self._pending_text,
                    self.state.in_code_block,
                    self.state.in_json_block,
                )

            if limit == 0:
                if self._pending_chunks:
                    next_text, next_dim = self._pending_chunks.pop(0)
                    if next_text:
                        self._pending_text += str(next_text)
                        if bool(next_dim) != self._pending_dim:
                            # 混合不同类型时，保守降级为非 dim，避免过度弱化输出。
                            self._pending_dim = False
                    continue
                return

            chunk_to_render = self._pending_text[:limit]

            # 2. 逐行处理
            self._process_content_lines(chunk_to_render, dim=self._pending_dim)

            # 3. 更新指针
            self._pending_text = self._pending_text[limit:]
            if not self._pending_text:
                self._pending_dim = False

    def _process_content_lines(self, content: str, *, dim: bool = False):
        """逐行解析并分发给 Painter 或 JsonProcessor"""
        lines = content.split("\n")

        for i, line in enumerate(lines):
            line = line.replace("\r", "")

            # 优先尝试匹配行末的 Fence (支持 "Text ```json" 格式)
            # 匹配: 任意前缀 + ``` + 可选语言标识(纯字母数字) + 可选空白
            fence_match = re.search(r"^(.*?)```([a-zA-Z0-9_]*)\s*$", line)
            if fence_match:
                pre_text = fence_match.group(1)
                lang = fence_match.group(2)

                # 构造 fence line
                fence_line = f"```{lang}"

                # 如果有前缀文本，先渲染前缀 (仅当不在块中时)
                if (
                    pre_text.strip()
                    and not self.state.in_code_block
                    and not self.state.in_json_block
                ):
                    self.painter.print_normal_line(
                        pre_text,
                        self.state.get_line_number(dim=dim),
                        dim=dim,
                    )
                    self.state.advance_line_number(dim=dim)

                # 调用处理逻辑
                self._handle_fence(fence_line, line, dim=dim)
                continue

            # 处理 Markdown Fence (```) - 保留旧逻辑作为兜底 (虽然上面的 regex 应该覆盖了大部分情况)
            stripped = line.strip()
            if stripped.startswith("```"):
                self._handle_fence(stripped, line, dim=dim)
                continue

            # 处理 JSON 块内容
            if self.state.in_json_block:
                if self.state.json_dim != bool(dim):
                    self.state.json_dim = bool(dim)
                self.json_processor.process_line(line, self.state, dim=self.state.json_dim)
                continue

            # 处理普通代码块内容
            if self.state.in_code_block:
                if self.state.code_dim != bool(dim):
                    self.state.code_dim = bool(dim)
                self.state.code_buffer.append(line)
                continue

            # 检测隐式 JSON 开始 (Heuristic)
            if self._try_detect_implicit_json(line, dim=dim):
                continue

            # 普通文本渲染
            if line.strip():  # 忽略纯空行，或者也可以渲染空行
                self.painter.print_normal_line(
                    line, self.state.get_line_number(dim=dim), dim=dim
                )
                self.state.advance_line_number(dim=dim)

    def _handle_fence(self, stripped_line: str, original_line: str, *, dim: bool = False):
        """处理 ``` 标记"""
        # 如果在隐式 JSON 中遇到 ```，这通常是 JSON 字符串的一部分，而不是 Markdown Fence
        if self.state.is_implicit_json:
            self.json_processor.process_line(
                original_line, self.state, dim=self.state.json_dim
            )
            return

        # 正常 Markdown Fence 逻辑
        if self.state.in_code_block or self.state.in_json_block:
            # 结束当前块
            if self.state.in_json_block:
                self.state.in_json_block = False
                self.state.json_dim = False
                logger.debug("显式 JSON 块结束")
            elif self.state.in_code_block:
                # 渲染累积的代码块
                full_code = "\n".join(self.state.code_buffer)
                consumed = self.painter.print_syntax_block(
                    full_code,
                    self.state.code_lang,
                    line_number_start=self.state.get_line_number(dim=self.state.code_dim),
                    dim=self.state.code_dim,
                )
                self.state.advance_line_number(
                    dim=self.state.code_dim, count=int(consumed or 0)
                )
                self.state.code_buffer.clear()
                self.state.in_code_block = False
                self.state.code_dim = False
                logger.debug("普通代码块结束")
        else:
            # 开始新块
            lang = stripped_line[3:].strip()
            if lang in ("json", "output_json"):
                self.state.in_json_block = True
                self.state.is_implicit_json = False
                self.state.json_dim = bool(dim)
                logger.debug("进入显式 JSON 块")
            else:
                self.state.in_code_block = True
                self.state.code_lang = lang
                self.state.code_dim = bool(dim)
                logger.debug(f"进入普通代码块: {lang}")

            # 打印围栏线
            if lang not in ("json", "output_json"):
                pass  # self.painter.print_json_structure_line("    │")

    def _try_detect_implicit_json(self, line: str, *, dim: bool = False) -> bool:
        """尝试检测是否开始了一个隐式的 JSON 块"""
        # 简单启发式：如果是 { 开头，或者包含 {"code": ...
        if "{" not in line:
            return False

        json_start = line.find("{")
        candidate = line[json_start:].strip()

        # 必须像是一个对象的开始
        if candidate.startswith('{"') or candidate == "{":
            # 这是一个新 JSON
            # 1. 渲染该行前面的普通文本
            pre_text = line[:json_start]
            if pre_text.strip():
                self.painter.print_normal_line(
                    pre_text, self.state.get_line_number(dim=dim), dim=dim
                )
                self.state.advance_line_number(dim=dim)

            # 2. 切换状态
            self.state.in_json_block = True
            self.state.is_implicit_json = True
            self.state.brace_balance = 0
            self.state.json_dim = bool(dim)

            # 3. 处理该行剩下的部分
            json_part = line[json_start:]
            self.json_processor.process_line(
                json_part, self.state, dim=self.state.json_dim
            )
            return True

        return False

    def _flush_incomplete_blocks(self):
        """清理结束时未闭合的块"""
        if self.state.in_code_block and self.state.code_buffer:
            logger.warning("强制闭合代码块")
            full_code = "\n".join(self.state.code_buffer)
            consumed = self.painter.print_syntax_block(
                full_code,
                self.state.code_lang,
                line_number_start=self.state.get_line_number(dim=self.state.code_dim),
                dim=self.state.code_dim,
            )
            self.state.advance_line_number(
                dim=self.state.code_dim, count=int(consumed or 0)
            )
            self.state.code_buffer.clear()

    def _save_log(self):
        """保存日志到执行目录的 save 目录下，按 agent 和任务名组织"""
        try:
            save_dir = get_stream_save_dir()
            save_dir.mkdir(parents=True, exist_ok=True)

            def _safe_part(s: str) -> str:
                s2 = (s or "").strip()
                if not s2:
                    return ""
                return re.sub(r"[^A-Za-z0-9_.-]+", "_", s2)[:80]

            run_ts = time.strftime("%Y%m%d_%H%M%S")
            op_part = _safe_part(self.op_name)
            agent_part = _safe_part(self.agent_name) or "agent"
            filename = f"{run_ts}_{op_part + '_' if op_part else ''}{agent_part}.txt"

            filepath = save_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(self.full_buffer)

            logger.debug(f"保存响应到: {filepath}")
        except Exception as e:
            log.warning("[StreamRenderer] save_log failed", exc_info=e)
            logger.warning(f"保存日志失败: {e}")
