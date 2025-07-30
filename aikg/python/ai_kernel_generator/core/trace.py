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
from dataclasses import dataclass


@dataclass
class AgentRecord:
    """Agent记录类，存储原始数据"""
    agent_name: str
    result: str = ""
    prompt: str = ""
    reasoning: str = ""
    error_log: str = ""
    profile: str = ""


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

    def save_parameters_to_files(self, agent_name: str, params: list):
        """保存参数到文件"""
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.op_name)
        os.makedirs(target_dir, exist_ok=True)

        base_name = f"I{self.task_id}_S{len(self.trace_list):02d}_{self.op_name}_{agent_name}_"

        for param_name, content in params:
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))

    def insert_agent_record(self, agent_name: str, result: str = "", prompt: str = "", reasoning: str = "",
                            error_log: str = "", profile: str = "") -> None:
        """
        插入agent执行记录（只保存原始数据，不进行解析）

        Args:
            agent_name: agent名称（designer, coder, verifier）
            result: 执行结果（原始json字符串）
            prompt: 使用的prompt
            reasoning: 推理过程
            error_log: 错误日志（主要用于verifier）
            profile: 性能数据
        """
        record = AgentRecord(
            agent_name=agent_name,
            result=result,
            prompt=prompt,
            reasoning=reasoning,
            error_log=error_log,
            profile=profile
        )
        self.trace_list.append(record)

        # 对于所有agent，保存原始json数据
        if agent_name in ["designer", "coder"]:
            self.save_parameters_to_files(agent_name, [
                ('result', result),  # 保存原始json
                ('prompt', prompt),
                ('reasoning', reasoning)
            ])
        elif agent_name == "verifier":
            # verifier记录错误日志
            if error_log:
                self.save_parameters_to_files(agent_name, [
                    ('error_log', error_log)
                ])

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
        base_name = f"I{self.task_id}_S{len(self.trace_list):02d}_{self.op_name}_{agent_name}_parsed.txt"
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
        base_name = f"I{self.task_id}_S{len(self.trace_list):02d}_{self.op_name}_{agent_name}_decision_"

        params = [
            ('result', res),
            ('prompt', prompt),
            ('reasoning', reasoning)
        ]
        for param_name, content in params:
            if not content:
                continue
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))

    def get_last_verifier_result(self) -> bool:
        """
        获取最后的verifier结果（简单的结果获取，不涉及解析）

        Returns:
            True表示验证成功，False表示验证失败
        """
        if not self.trace_list:
            return False

        # 从后向前查找verifier记录
        for record in reversed(self.trace_list):
            if record.agent_name == "verifier":
                result = record.result
                if result == "True" or result is True:
                    return True
                elif result == "False" or result is False:
                    return False

        return False

    def get_last_verifier_error_log(self) -> str:
        """
        获取最后的verifier错误日志

        Returns:
            错误日志字符串
        """
        if not self.trace_list:
            return ""

        # 从后向前查找verifier记录
        for record in reversed(self.trace_list):
            if record.agent_name == "verifier":
                return record.error_log or ""

        return ""

    def find_last_agent_record(self, agent_names: list):
        """
        查找最后的agent记录（返回完整记录，供conductor使用）

        Args:
            agent_names: agent名字列表

        Returns:
            AgentRecord或None
        """
        if not agent_names or not self.trace_list:
            return None

        # 从后向前查找
        for record in reversed(self.trace_list):
            if record.agent_name in agent_names:
                return record

        return None
