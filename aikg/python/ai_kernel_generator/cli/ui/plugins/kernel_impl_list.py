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

"""Kernel Impl List panel plugin"""

from typing import List, Tuple

from ai_kernel_generator.cli.ui.panel_plugin import PanelPlugin

class KernelImplListPlugin(PanelPlugin):
    """Kernel Impl List plugin - displays current task and implementation history"""

    def __init__(self):
        # 当前任务状态
        self.current_task = {
            "task_name": "",
            "phase": "",  # subagent name
        }

        # 实现历史（按 speedup 降序）
        self.history: List[dict] = []
        # 每个历史项: {"speedup": float, "gen_time": float, "base_time": float, "log_dir": str}

    def get_name(self) -> str:
        """返回插件名称"""
        return "kernel_impl_list"

    def render_current_task_fragments(self, width: int) -> List[Tuple[str, str]]:
        """渲染当前任务部分（不带边框，边框由 Frame 组件处理）"""
        fragments = []
        content_width = max(1, width)

        task_name = self.current_task.get("task_name", "")
        phase = self.current_task.get("phase", "")
        if task_name or phase:
            parts = []
            if task_name:
                parts.append(f"Op: {task_name}")
            if phase:
                parts.append(f"Phase: {phase}")
            content_line = " ".join(parts)
            # 确保行长度不超过宽度
            if len(content_line) > content_width:
                content_line = content_line[:content_width - 3] + "..."
            fragments.append(("class:panel.body", content_line))
        else:
            fragments.append(("class:panel.body", "等待任务启动..."))
        return fragments

    def render_history_fragments(self, width: int) -> List[Tuple[str, str]]:
        """渲染历史部分（不带边框，边框由 Frame 组件处理）"""
        fragments = []
        content_width = max(1, width)

        if self.history:
            history_items = self.history[:5]
            for idx, item in enumerate(history_items, 1):
                speedup = item.get("speedup", 0.0)
                gen_time = item.get("gen_time", 0.0)
                base_time = item.get("base_time", 0.0)
                log_dir = item.get("log_dir", "")

                # 格式化时间显示（微秒）
                gen_time_str = f"{gen_time:.0f}µs" if gen_time > 0 else "N/A"
                base_time_str = f"{base_time:.0f}µs" if base_time > 0 else "N/A"
                speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
                # 处理 log_dir 显示
                log_dir_display = log_dir
                if log_dir:
                    # 如果路径太长，截断显示
                    max_log_dir_len = content_width - 50  # 为其他内容预留空间
                    if len(log_dir) > max_log_dir_len:
                        log_dir_display = "..." + log_dir[-(max_log_dir_len - 3):]
                else:
                    log_dir_display = "N/A"

                line = f"Speedup: {speedup_str} | Gen: {gen_time_str} | Base: {base_time_str} | Log: {log_dir_display}"
                # 确保行长度不超过宽度
                if len(line) > content_width:
                    line = line[:content_width - 3] + "..."
                # 如果不是最后一条，添加换行符
                if idx < len(history_items):
                    line += "\n"
                fragments.append(("class:panel.body", line))
        else:
            fragments.append(("class:panel.body", "暂无历史记录"))
        return fragments

    def render_fragments(self, width: int) -> List[Tuple[str, str]]:
        """渲染面板内容（兼容旧接口，返回合并后的内容）"""
        fragments = []
        content_width = max(1, width)

        # 上模块：当前任务
        fragments.extend(self.render_current_task_fragments(content_width))

        # 下模块：历史 Top 5
        fragments.extend(self.render_history_fragments(content_width))
        return fragments

    def on_data_update(self, data: dict) -> bool:
        """更新数据，返回是否有实际变化"""
        action = data.get("action")
        changed = False
        
        if action == "update_current":
            # 更新当前任务
            update_data = data.get("data", {})
            new_task = {
                "task_name": update_data.get("task_name", ""),
                "phase": update_data.get("phase", ""),
            }
            # 检查是否有变化
            if new_task != self.current_task:
                self.current_task = new_task
                changed = True
        elif action == "move_to_history":
            # 将当前任务移到历史（只要有性能数据就添加，不检查状态）
            history_data = data.get("data", {})
            speedup = float(history_data.get("speedup", 0))
            gen_time = float(history_data.get("gen_time", 0))
            base_time = float(history_data.get("base_time", 0))
            log_dir = history_data.get("log_dir", "")
            # 只要有性能数据就添加到历史（允许 speedup < 1.0）
            if speedup != 0.0 or gen_time != 0.0 or base_time != 0.0:
                history_item = {
                    "speedup": speedup,
                    "gen_time": gen_time,
                    "base_time": base_time,
                    "log_dir": log_dir,
                }
                self.history.append(history_item)
                self.history.sort(key=lambda x: x.get("speedup", 0), reverse=True)
                self.history = self.history[:5]
                changed = True
                # 保留当前任务显示
        elif action == "reset":
            # 检查是否有变化
            if self.current_task.get("task_name") or self.current_task.get("phase") or self.history:
                changed = True
            # 重置状态
            self.current_task = {
                "task_name": "",
                "phase": "",
            }
            self.history = []
        
        return changed
