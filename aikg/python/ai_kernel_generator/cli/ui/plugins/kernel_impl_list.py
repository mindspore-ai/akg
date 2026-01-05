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
            "phase": "",  # subagent_coderonly/verifier/conductor
            "status": "",  # running/done
            "save_path": "",
        }

        # 实现历史（按 speedup 降序）
        self.history: List[dict] = []
        # 每个历史项: {"round": int, "speedup": float, "gen_time": float, "base_time": float, "save_path": str}

    def get_name(self) -> str:
        """返回插件名称"""
        return "kernel_impl_list"

    def render_fragments(self, width: int) -> List[Tuple[str, str]]:
        """渲染面板内容"""
        fragments = []
        sep = "─" * max(1, width)

        # 上模块：当前任务
        fragments.append(("class:panel.separator", sep + "\n"))
        fragments.append(("class:panel.title", "当前任务\n"))

        if self.current_task.get("task_name") or self.current_task.get("save_path"):
            # 只显示 op_name 和 save_path
            save_path = self.current_task.get("save_path", "")
            task_name = self.current_task.get("task_name", "")

            if task_name:
                fragments.append(("class:panel.body", f"Op: {task_name}\n"))
            if save_path:
                # 如果路径太长，截断显示
                if len(save_path) > width - 10:
                    display_path = "..." + save_path[-(width - 13) :]
                else:
                    display_path = save_path
                fragments.append(("class:panel.body", f"Save: {display_path}\n"))
        else:
            fragments.append(("class:panel.body", "等待任务启动...\n"))

        # 分隔线
        fragments.append(("class:panel.separator", sep + "\n"))

        # 下模块：历史 Top 5
        fragments.append(("class:panel.title", "实现历史 Top 5\n"))

        if self.history:
            for idx, item in enumerate(self.history[:5], 1):
                round_num = item.get("round", idx)
                speedup = item.get("speedup", 0.0)
                gen_time = item.get("gen_time", 0.0)
                base_time = item.get("base_time", 0.0)

                # 格式化时间显示（微秒）
                gen_time_str = f"{gen_time:.0f}µs" if gen_time > 0 else "N/A"
                base_time_str = f"{base_time:.0f}µs" if base_time > 0 else "N/A"
                speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"

                line = f"#{round_num} Speedup: {speedup_str} | Gen: {gen_time_str} | Base: {base_time_str}"
                # 如果行太长，截断
                if len(line) > width:
                    line = line[: width - 3] + "..."
                fragments.append(("class:panel.body", line + "\n"))
        else:
            fragments.append(("class:panel.body", "暂无历史记录\n"))

        fragments.append(("class:panel.separator", sep))
        return fragments

    def on_data_update(self, data: dict) -> None:
        """更新数据"""
        action = data.get("action")
        if action == "update_current":
            # 更新当前任务
            update_data = data.get("data", {})
            self.current_task = {
                "task_name": update_data.get("task_name", ""),
                "save_path": update_data.get("save_path", ""),
            }
        elif action == "move_to_history":
            # 将当前任务移到历史（只要有性能数据就添加，不检查状态）
            history_data = data.get("data", {})
            speedup = float(history_data.get("speedup", 0))
            gen_time = float(history_data.get("gen_time", 0))
            base_time = float(history_data.get("base_time", 0))
            
            # 只要有性能数据就添加到历史（允许 speedup < 1.0）
            if speedup != 0.0 or gen_time != 0.0 or base_time != 0.0:
                history_item = {
                    "round": len(self.history) + 1,
                    "speedup": speedup,
                    "gen_time": gen_time,
                    "base_time": base_time,
                    "save_path": self.current_task.get("save_path", ""),
                }
                self.history.append(history_item)
                self.history.sort(key=lambda x: x.get("speedup", 0), reverse=True)
                self.history = self.history[:5]
                # 保留当前任务显示
        elif action == "reset":
            # 重置状态
            self.current_task = {
                "task_name": "",
                "save_path": "",
            }
            self.history = []
