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

"""
大模型推理痕迹追踪模块
只负责存储原始数据，不进行解析逻辑
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentRecord:
    """Agent记录类，存储原始数据"""
    agent_name: str
    result: str = ""
    prompt: str = ""
    reasoning: str = ""
    error_log: str = ""
    profile_res: dict = field(default_factory=dict)


class Trace:
    """
    大模型推理痕迹追踪类
    职责：存储原始数据，保存文件，不负责解析逻辑
    """

    def __init__(self, op_name, task_id, log_dir: str):
        self.op_name = op_name
        self.task_id = task_id
        self.log_dir = log_dir
        self.trace_list = []  # 存储AgentRecord记录

    def get_step_count(self) -> int:
        """获取当前步骤计数（已记录的 trace_list 条数）"""
        return len(self.trace_list)
    
    def get_log_task_id(self) -> str:
        """获取日志文件名中使用的 task_id"""
        return self.task_id

    def save_parameters_to_files(self, agent_name: str, params: list):
        """保存参数到文件"""
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.op_name)
        os.makedirs(target_dir, exist_ok=True)

        base_name = f"Iteration{self.task_id}_Step{len(self.trace_list):02d}_{self.op_name}_{agent_name}_"

        for param_name, content in params:
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))

    def insert_agent_record(self, agent_name: str, result: str = "", prompt: str = "", reasoning: str = "",
                            error_log: str = "", profile_res: Optional[dict] = None,
                            session_id: Optional[str] = None, elapsed_s: Optional[float] = None) -> None:
        """
        插入agent执行记录（只保存原始数据，不进行解析）

        Args:
            agent_name: agent名称（designer, coder, verifier）
            result: 执行结果（原始json字符串）
            prompt: 使用的prompt
            reasoning: 推理过程
            error_log: 错误日志（主要用于verifier）
            profile_res: 性能数据字典
            session_id: 可选，用于向 CLI WS session 推送 DisplayMessage
            elapsed_s: 可选，调用方计时后传入，用于展示耗时
        """
        if profile_res is None:
            profile_res = {}
        
        record = AgentRecord(
            agent_name=agent_name,
            result=result,
            prompt=prompt,
            reasoning=reasoning,
            error_log=error_log,
            profile_res=profile_res
        )
        self.trace_list.append(record)

        # 对于所有agent，保存原始json数据
        wrote_files = False
        if agent_name in ["designer", "coder", "kernel_gen", "sketch", "test_case_generator"]:
            self.save_parameters_to_files(agent_name, [
                ('result', result),  # 保存原始json
                ('prompt', prompt),
                ('reasoning', reasoning)
            ])
            wrote_files = True
        elif agent_name == "verifier":
            # verifier记录错误日志
            if error_log:
                self.save_parameters_to_files(agent_name, [
                    ('error_log', error_log)
                ])
            # verifier 总是发送消息（包含结果和 profile 数据）
            self._send_verifier_result_message(
                session_id=session_id,
                elapsed_s=elapsed_s,
                verify_result=result,
                error_log=error_log,
                profile_res=profile_res,
            )
            return  # verifier 使用专用方法发送消息，不走通用逻辑

        # 当且仅当本次确实写出了 trace 文件时，发送 related_files 消息（每条消息一个框）
        if wrote_files:
            self._send_related_files_message(
                agent_name=agent_name,
                session_id=session_id,
                elapsed_s=elapsed_s,
            )

    def _send_verifier_result_message(
        self,
        session_id: Optional[str],
        elapsed_s: Optional[float],
        verify_result: str,
        error_log: str,
        profile_res: dict,
    ) -> None:
        """
        发送 verifier 结果消息，包含验证结果和 profile 数据。

        仅用于 CLI 展示，不影响主流程；失败时吞掉异常。
        """
        try:
            session_id = str(session_id or "").strip()
            if not session_id:
                return

            # 延迟导入，避免 core 层引入过重的 CLI 依赖
            from akg_agents.cli.runtime.message_sender import send_message
            from akg_agents.cli.messages import DisplayMessage

            # 构建结果状态
            is_passed = verify_result.lower() == "true"
            status = "✅ PASSED" if is_passed else "❌ FAILED"

            # 组装消息（每个信息一行，没有内容则不添加）
            if elapsed_s is not None:
                lines = [f"◀ verifier done in {elapsed_s:.2f}s, status: {status}"]
            else:
                lines = [f"◀ verifier status: {status}"]
            
            # profile 信息（如果有）
            if profile_res and is_passed:
                lines.append(f"  profile: {profile_res}")
            
            # 相关文件路径（如果有 error_log）
            if error_log and self.log_dir:
                expanded_log_dir = os.path.expanduser(str(self.log_dir))
                target_dir = os.path.join(expanded_log_dir, self.op_name)
                step_index = len(self.trace_list)
                prefix = f"Iteration{self.task_id}_Step{step_index:02d}_{self.op_name}_verifier_"
                glob_path = os.path.abspath(os.path.join(target_dir, prefix + "*"))
                lines.append(f"  realated_files: {glob_path}")

            text = "\n".join(lines)

            send_message(session_id, DisplayMessage(text=text))
        except Exception:
            # 过程展示失败不应影响主流程
            return

    def _send_related_files_message(self, agent_name: str, session_id: Optional[str], elapsed_s: Optional[float]) -> None:
        """
        发送包含相关落盘文件 glob 的 DisplayMessage。

        仅用于 CLI 展示，不影响主流程；失败时吞掉异常。
        """
        try:
            session_id = str(session_id or "").strip()
            if not session_id:
                return

            # log_dir 为空时不发送（避免无意义路径）
            if not self.log_dir:
                return

            expanded_log_dir = os.path.expanduser(str(self.log_dir))
            target_dir = os.path.join(expanded_log_dir, self.op_name)
            step_index = len(self.trace_list)  # append 后，需与 save_parameters_to_files 的 Sxx 一致
            prefix = f"Iteration{self.task_id}_Step{step_index:02d}_{self.op_name}_{agent_name}_"
            glob_path = os.path.abspath(os.path.join(target_dir, prefix + "*"))

            # 延迟导入，避免 core 层引入过重的 CLI 依赖
            from akg_agents.cli.runtime.message_sender import send_message
            from akg_agents.cli.messages import DisplayMessage

            # 组装消息（换行显示）
            # 注意：这里沿用历史拼写 realated_files（避免下游依赖破坏）
            if elapsed_s is not None:
                text = f"◀ {agent_name} done in {elapsed_s:.2f}s\n  realated_files: {glob_path}"
            else:
                text = f"◀ {agent_name} done\n  realated_files: {glob_path}"

            send_message(session_id, DisplayMessage(text=text))
        except Exception:
            # 过程展示失败不应影响主流程
            return

    def save_parsed_code(self, agent_name: str, params: list) -> None:
        """
        保存解析后的内容到一个文件（由conductor调用）

        Args:
            agent_name: agent名称
            params: 参数列表，格式为 [('参数名', '内容'), ...]
        """
        if not params:
            return

        # 保存到recorder目录，与trace记录分隔开
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.op_name, "recorder")
        os.makedirs(target_dir, exist_ok=True)

        # 生成单个文件
        base_name = f"Iteration{self.task_id}_Step{len(self.trace_list):02d}_{self.op_name}_{agent_name}_parsed.txt"
        file_path = os.path.join(target_dir, base_name)

        # 将所有内容合并到一个文件，每个部分用几个回车隔开
        content_parts = []
        for param_name, content in params:
            if content:
                content_parts.append(f"=== {param_name} ===\n{content}")

        if content_parts:
            combined_content = "\n\n\n".join(content_parts)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(combined_content)

    def log_record(self, record_name: str, params: list,
                   subdirectory: str = None) -> int:
        """
        记录一个工作流步骤，并将参数保存为独立文件。
        
        与 TraceSystem.log_record 接口一致，用于 nodes.py 等调用方
        在不区分 Trace / TraceSystem 的情况下统一调用。
        
        文件命名:
            Iteration{task_id}_Step{NN:02d}_{op_name}_{record_name}_{param_name}.txt
        
        Args:
            record_name: 记录名称
            params: 参数列表，格式为 [('param_name', 'content'), ...]
            subdirectory: 可选的子目录名（在 {log_dir}/{op_name}/ 之下）
        
        Returns:
            当前步骤号（自增后）
        """
        self.trace_list.append(AgentRecord(agent_name=record_name))
        step = len(self.trace_list)
        
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.op_name)
        if subdirectory:
            target_dir = os.path.join(target_dir, subdirectory)
        os.makedirs(target_dir, exist_ok=True)
        
        base_name = f"Iteration{self.task_id}_Step{step:02d}_{self.op_name}_{record_name}_"
        
        for param_name, content in params:
            if content is None:
                continue
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))
        
        return step

    def log_merged_record(self, record_name: str, params: list,
                          filename_suffix: str = "_parsed.txt",
                          subdirectory: str = "recorder") -> None:
        """
        将多个参数合并保存到单个文件（不自增步骤计数器）。
        
        与 TraceSystem.log_merged_record 接口一致。
        
        Args:
            record_name: 记录名称
            params: 参数列表，格式为 [('section_name', 'content'), ...]
            filename_suffix: 文件名后缀（默认 "_parsed.txt"）
            subdirectory: 子目录名（默认 "recorder"）
        """
        if not params:
            return
        
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.op_name, subdirectory)
        os.makedirs(target_dir, exist_ok=True)
        
        step = len(self.trace_list)
        base_name = f"Iteration{self.task_id}_Step{step:02d}_{self.op_name}_{record_name}{filename_suffix}"
        file_path = os.path.join(target_dir, base_name)
        
        content_parts = []
        for section_name, content in params:
            if content:
                content_parts.append(f"=== {section_name} ===\n{content}")

        if content_parts:
            combined_content = "\n\n\n".join(content_parts)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(combined_content)

    def insert_conductor_agent_record(self, res: str, prompt: str, reasoning: str, agent_name: str) -> None:
        """
        插入conductor相关的记录（包括LLM决策、检查等）

        Args:
            res: 结果内容
            prompt: prompt内容
            reasoning: 推理过程
            agent_name: agent名称或操作类型（如decision、check、analyze等）
        """
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.op_name, "conductor")
        os.makedirs(target_dir, exist_ok=True)

        # 直接使用agent_name，不进行转换
        base_name = f"Iteration{self.task_id}_Step{len(self.trace_list):02d}_{self.op_name}_{agent_name}_decision_"

        params = [
            ('result', res),
            ('prompt', prompt),
            ('reasoning', reasoning)
        ]
        for param_name, content in params:
            if content is None:
                continue
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))
