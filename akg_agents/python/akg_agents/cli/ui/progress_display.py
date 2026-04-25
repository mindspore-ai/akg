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
Adaptive Search Progress Display - 美观的进度显示组件

使用 Rich Live 实现实时原地更新的状态栏，避免重复打印产生的视觉混乱。

设计特点：
- 实时原地更新（不滚动终端）
- 分层信息显示（整体进度 + 任务详情 + 性能数据）
- 平滑动画效果
- 支持在 prompt_toolkit TUI 中集成
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.live import Live
from rich.style import Style

logger = logging.getLogger(__name__)


_DEFAULT_ETA_PER_TASK_S = 120


def _format_elapsed_and_eta(elapsed_s: int, completed: int, total: int) -> str:
    """格式化已用时间和预计剩余时间
    
    已用时: 秒为单位
    预计剩余: 分钟为单位（向上取整），尚无完成任务时按默认值估算
    """
    if completed > 0 and completed < total:
        avg_per_task = elapsed_s / completed
        remaining_tasks = total - completed
        eta_min = _seconds_to_ceil_minutes(avg_per_task * remaining_tasks)
        return f"已用时:{elapsed_s}s, 预计剩余:{eta_min}min"
    elif completed == 0 and total > 0:
        eta_min = _seconds_to_ceil_minutes(_DEFAULT_ETA_PER_TASK_S * total)
        return f"已用时:{elapsed_s}s, 预计剩余:~{eta_min}min"
    
    return f"已用时:{elapsed_s}s"


def _seconds_to_ceil_minutes(seconds: float) -> int:
    """将秒数向上取整为分钟"""
    import math
    return max(1, math.ceil(seconds / 60))


@dataclass
class TaskPerformance:
    """任务性能记录"""
    task_id: str
    gen_time: float = 0.0  # 微秒
    speedup: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class AdaptiveSearchProgressState:
    """自适应搜索进度状态"""
    op_name: str = ""
    max_total_tasks: int = 0
    total_submitted: int = 0
    total_completed: int = 0
    total_success: int = 0
    total_failed: int = 0
    running_count: int = 0
    waiting_count: int = 0
    start_time: Optional[datetime] = None
    
    # 最佳性能记录
    best_performances: List[TaskPerformance] = field(default_factory=list)
    
    # 最近完成的任务
    recent_task_id: str = ""
    recent_success: bool = False
    recent_gen_time: float = 0.0
    
    def update(self, data: Dict[str, Any]) -> None:
        """从消息数据更新状态"""
        self.op_name = data.get("op_name", self.op_name)
        self.max_total_tasks = data.get("max_total_tasks", self.max_total_tasks)
        self.total_submitted = data.get("total_submitted", self.total_submitted)
        self.total_completed = data.get("total_completed", self.total_completed)
        self.total_success = data.get("total_success", self.total_success)
        self.total_failed = data.get("total_failed", self.total_failed)
        self.running_count = data.get("running_count", self.running_count)
        self.waiting_count = data.get("waiting_count", self.waiting_count)
    
    def add_success_task(self, task_id: str, gen_time: float, speedup: float) -> None:
        """添加成功的任务记录"""
        self.recent_task_id = task_id
        self.recent_success = True
        self.recent_gen_time = gen_time
        
        # 更新最佳性能列表
        perf = TaskPerformance(task_id=task_id, gen_time=gen_time, speedup=speedup)
        self.best_performances.append(perf)
        # 保持前5个最佳性能
        self.best_performances.sort(key=lambda x: x.gen_time)
        self.best_performances = self.best_performances[:5]
    
    def get_elapsed_seconds(self) -> int:
        """获取已用时间（秒）"""
        if self.start_time:
            return int((datetime.now() - self.start_time).total_seconds())
        return 0
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_completed == 0:
            return 0.0
        return self.total_success / self.total_completed


class AdaptiveSearchProgressDisplay:
    """
    自适应搜索进度显示器
    
    使用 Rich Live 组件创建美观的实时进度显示。
    设计为可在 prompt_toolkit TUI 底部显示。
    """
    
    def __init__(
        self, 
        op_name: str = "",
        max_tasks: int = 100,
        console: Optional[Console] = None,
    ):
        self.state = AdaptiveSearchProgressState(
            op_name=op_name,
            max_total_tasks=max_tasks,
            start_time=datetime.now(),
        )
        self.console = console or Console()
        self._spinner_frames = ["⣾", "⣽", "⣻", "⢿", "⡿", "⣟", "⣯", "⣷"]
        self._spinner_idx = 0
    
    def update(self, data: Dict[str, Any]) -> None:
        """更新进度数据"""
        self.state.update(data)
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_frames)
    
    def add_success(self, task_id: str, gen_time: float, speedup: float) -> None:
        """添加成功任务"""
        self.state.add_success_task(task_id, gen_time, speedup)
    
    def render(self) -> Panel:
        """渲染进度面板"""
        # 构建内容组
        elements = []
        
        # 1. 标题行：操作名称 + spinner
        spinner = self._spinner_frames[self._spinner_idx]
        title_text = Text()
        title_text.append(f"{spinner} ", style="bold cyan")
        title_text.append("Adaptive Search: ", style="bold")
        title_text.append(self.state.op_name, style="bold green")
        elements.append(title_text)
        
        # 2. 进度条
        progress_bar = self._render_progress_bar()
        elements.append(progress_bar)
        
        # 3. 统计卡片
        stats_table = self._render_stats_table()
        elements.append(stats_table)
        
        # 4. 最佳性能（如果有）
        if self.state.best_performances:
            best_text = self._render_best_performance()
            elements.append(best_text)
        
        # 组装面板
        content = Group(*elements)
        
        return Panel(
            content,
            border_style="cyan",
            padding=(0, 1),
        )
    
    def _render_progress_bar(self) -> Text:
        """渲染进度条"""
        completed = self.state.total_completed
        total = self.state.max_total_tasks
        
        if total <= 0:
            return Text("Preparing...", style="dim")
        
        # 进度条宽度
        bar_width = 40
        filled = int(bar_width * completed / total) if total > 0 else 0
        empty = bar_width - filled
        
        # 构建进度条
        bar = Text()
        bar.append("  ")
        bar.append("━" * filled, style="bold cyan")
        bar.append("━" * empty, style="dim")
        bar.append(f"  {completed}/{total} tasks", style="bold")
        
        return bar
    
    def _render_stats_table(self) -> Table:
        """渲染统计卡片"""
        table = Table(show_header=False, show_edge=False, pad_edge=False, padding=(0, 2))
        
        # 成功
        success_text = Text()
        success_text.append("✅ ", style="green")
        success_text.append(f"成功 {self.state.total_success}", style="bold green")
        table.add_column(justify="left")
        
        # 失败
        fail_text = Text()
        fail_text.append("❌ ", style="red")
        fail_text.append(f"失败 {self.state.total_failed}", style="bold red")
        table.add_column(justify="left")
        
        # 运行中
        running_text = Text()
        running_text.append("🔄 ", style="yellow")
        running_text.append(f"运行 {self.state.running_count}", style="bold yellow")
        table.add_column(justify="left")
        
        # 耗时
        elapsed = self.state.get_elapsed_seconds()
        time_text = Text()
        time_text.append("⏱️ ", style="blue")
        time_text.append(f"{elapsed}s", style="bold blue")
        table.add_column(justify="left")
        
        table.add_row(success_text, fail_text, running_text, time_text)
        
        return table
    
    def _render_best_performance(self) -> Text:
        """渲染最佳性能信息"""
        if not self.state.best_performances:
            return Text()
        
        best = self.state.best_performances[0]
        
        text = Text()
        text.append("\n  ")
        text.append("🏆 ", style="gold1")
        text.append("最佳: ", style="bold")
        text.append(best.task_id[:16], style="cyan")
        text.append(f" ({best.gen_time:.2f}us", style="green")
        text.append(f", {best.speedup:.2f}x 加速)", style="bold green")
        
        return text
    
    def render_compact(self) -> str:
        """渲染紧凑格式（用于 spinner 状态栏）"""
        s = self.state
        elapsed = s.get_elapsed_seconds()
        
        if s.max_total_tasks > 0:
            progress = f"Task {s.total_completed}/{s.max_total_tasks}"
            parts = [progress]
            if s.total_success > 0:
                parts.append(f"成功:{s.total_success}")
            if s.total_failed > 0:
                parts.append(f"失败:{s.total_failed}")
            if s.running_count > 0:
                parts.append(f"进行中:{s.running_count}")
            parts.append(_format_elapsed_and_eta(elapsed, s.total_completed, s.max_total_tasks))
            return " | ".join(parts)
        else:
            return f"运行中... ({elapsed}s)"
    
    def get_spinner_text(self) -> str:
        """获取带 spinner 的状态文本"""
        spinner = self._spinner_frames[self._spinner_idx]
        return f"{spinner} {self.render_compact()}"


class EvolveProgressState:
    """Evolve 进度状态"""
    
    def __init__(self):
        self.op_name: str = ""
        self.current_round: int = 0
        self.max_rounds: int = 0
        self.parallel_num: int = 0
        self.total_tasks: int = 0
        self.total_completed: int = 0
        self.total_success: int = 0
        self.total_failed: int = 0
        self.running_count: int = 0
        self.phase: str = ""
        self.start_time: Optional[datetime] = None
    
    def update(self, data: Dict[str, Any]) -> None:
        self.op_name = data.get("op_name", self.op_name)
        self.current_round = data.get("current_round", self.current_round)
        self.max_rounds = data.get("max_rounds", self.max_rounds)
        self.parallel_num = data.get("parallel_num", self.parallel_num)
        self.total_tasks = data.get("total_tasks", self.total_tasks)
        self.total_completed = data.get("total_completed", self.total_completed)
        self.total_success = data.get("total_success", self.total_success)
        self.total_failed = data.get("total_failed", self.total_failed)
        self.running_count = data.get("running_count", self.running_count)
        self.phase = data.get("phase", self.phase)
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def get_elapsed_seconds(self) -> int:
        if self.start_time:
            return int((datetime.now() - self.start_time).total_seconds())
        return 0
    
    def render_compact(self) -> str:
        """渲染紧凑的 evolve 进度字符串"""
        elapsed = self.get_elapsed_seconds()
        
        round_str = f"Round {self.current_round}/{self.max_rounds}"
        task_str = f"Task {self.total_completed}/{self.total_tasks}"
        
        parts = [round_str, task_str]
        if self.total_success > 0:
            parts.append(f"成功:{self.total_success}")
        if self.total_failed > 0:
            parts.append(f"失败:{self.total_failed}")
        if self.running_count > 0:
            parts.append(f"进行中:{self.running_count}")
        parts.append(_format_elapsed_and_eta(elapsed, self.total_completed, self.total_tasks))
        
        return " | ".join(parts)


class AdaptiveSearchLiveDisplay:
    """
    基于 Rich Live 的实时进度显示器
    
    在单独的 Live 区域中显示进度，适合在 terminal 底部显示。
    """
    
    def __init__(
        self,
        op_name: str = "",
        max_tasks: int = 100,
        console: Optional[Console] = None,
        refresh_per_second: int = 4,
    ):
        self.display = AdaptiveSearchProgressDisplay(
            op_name=op_name,
            max_tasks=max_tasks,
            console=console,
        )
        self.refresh_per_second = refresh_per_second
        self._live: Optional[Live] = None
    
    def start(self) -> None:
        """启动 Live 显示"""
        if self._live is not None:
            return
        
        self._live = Live(
            self.display.render(),
            console=self.display.console,
            refresh_per_second=self.refresh_per_second,
            transient=True,  # 完成后清除
        )
        self._live.start()
    
    def stop(self) -> None:
        """停止 Live 显示"""
        if self._live is not None:
            self._live.stop()
            self._live = None
    
    def update(self, data: Dict[str, Any]) -> None:
        """更新进度并刷新显示"""
        self.display.update(data)
        if self._live is not None:
            self._live.update(self.display.render())
    
    def __enter__(self) -> "AdaptiveSearchLiveDisplay":
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()