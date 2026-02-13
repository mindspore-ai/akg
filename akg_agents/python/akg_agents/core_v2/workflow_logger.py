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

"""工作流日志记录器

统一的日志记录类，用于记录 workflow 中各节点产生的中间文件。
每个 workflow 实例持有一个独立的 WorkflowLogger，互不干扰。

职责：
- 将各 Agent 节点的 prompt、result、reasoning 等保存为独立文件
- 管理步骤计数器（step_count），保证文件编号的连续性
- 合并保存解析后的结构化内容

设计原则：
- 通用接口，不绑定任何特定 Agent 或领域逻辑
- 每个实例独立计数，天然支持并发 workflow（如 evolve、adaptive_search）
- 接口与旧 Trace 类兼容（duck-typing），便于在 nodes.py、routers.py 等
  调用方中无差别使用

使用场景：
1. KernelGenOnlyWorkflow（kernel_agent 调用）：
   由 OpBaseWorkflow.prepare_config() 创建并注入到 workflow_resources["trace"]
2. Evolve / Adaptive Search 内部的 LangGraphTask：
   在 LangGraphTask.__init__() 中创建，替代旧 Trace 类
3. 原有脚本直接调用 LangGraphTask（非 kernel_agent 场景）：同 2

文件命名规则：
    Iteration{task_id}_Step{NN:02d}_{category}_{record_name}_{param_name}.txt

目录结构：
    {log_dir}/{category}/                     ← 主目录
    {log_dir}/{category}/{subdirectory}/      ← 可选子目录（如 conductor/）
    {log_dir}/{category}/recorder/            ← 合并文件默认目录
"""

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


class WorkflowLogger:
    """工作流日志记录器
    
    每个 workflow 实例持有独立的 WorkflowLogger，
    通过步骤计数器保证文件编号的连续性和唯一性。
    
    Attributes:
        log_dir: 日志文件根目录（如 node_00X/logs 或 akg_agents_logs/Task_xxx）
        category: 分类子目录名，也出现在文件名中（如算子名 "l1norm"）
        task_id: 日志文件名中的任务标识（如 "0", "Round1_Island0_Task0"）
    """
    
    def __init__(self, log_dir: str, category: str, task_id: str):
        """初始化工作流日志记录器
        
        Args:
            log_dir: 日志文件根目录
            category: 分类子目录名（也出现在文件名中）
            task_id: 任务标识（出现在文件名前缀 Iteration{task_id} 中）
        """
        self.log_dir = log_dir
        self.category = category
        self.task_id = task_id
        self._step_records: List[dict] = []
        
        logger.info(
            f"[WorkflowLogger] 初始化: log_dir={log_dir}, "
            f"category={category}, task_id={task_id}"
        )
    
    # ==================== 核心接口 ====================
    
    def log_record(self, record_name: str, params: list,
                   subdirectory: str = None) -> int:
        """记录一个工作流步骤，将参数保存为独立文件
        
        每调用一次，内部步骤计数器自增。每个 (param_name, content) 对
        保存为一个独立的 .txt 文件。
        
        文件命名:
            Iteration{task_id}_Step{NN:02d}_{category}_{record_name}_{param_name}.txt
        
        文件目录:
            {log_dir}/{category}/                    （无 subdirectory 时）
            {log_dir}/{category}/{subdirectory}/     （有 subdirectory 时）
        
        Args:
            record_name: 记录名称，用于文件名中标识来源
                        （如 "kernel_gen", "designer", "verifier"）
            params: 参数列表，格式为 [('param_name', content), ...]
                   content 为 None 时跳过该文件；空字符串 "" 会保存（生成空文件）
            subdirectory: 可选的子目录名（在 {log_dir}/{category}/ 之下）
                        （如 "conductor" → 文件放入 {category}/conductor/）
        
        Returns:
            当前步骤号（自增后）
        """
        # 即使 log_dir 为空，也自增步骤计数器以保持编号一致性
        self._step_records.append({"name": record_name})
        step = len(self._step_records)
        
        if not self.log_dir:
            logger.debug(
                f"[WorkflowLogger] log_record({record_name}): "
                f"log_dir 未设置，跳过文件保存"
            )
            return step
        
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.category)
        if subdirectory:
            target_dir = os.path.join(target_dir, subdirectory)
        os.makedirs(target_dir, exist_ok=True)
        
        base_name = (
            f"Iteration{self.task_id}_Step{step:02d}_"
            f"{self.category}_{record_name}_"
        )
        
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
        """将多个参数合并保存到单个文件（不自增步骤计数器）
        
        适用于将解析后的结构化内容集中存储。各部分用
        "=== section_name ===" 分隔。
        
        文件命名:
            Iteration{task_id}_Step{NN:02d}_{category}_{record_name}{filename_suffix}
        
        文件目录:
            {log_dir}/{category}/{subdirectory}/
        
        Args:
            record_name: 记录名称
            params: 参数列表，格式为 [('section_name', content), ...]
                   空内容的 section 会被跳过
            filename_suffix: 文件名后缀（默认 "_parsed.txt"）
            subdirectory: 子目录名（默认 "recorder"）
        """
        if not self.log_dir or not params:
            return
        
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.category, subdirectory)
        os.makedirs(target_dir, exist_ok=True)
        
        step = len(self._step_records)
        base_name = (
            f"Iteration{self.task_id}_Step{step:02d}_"
            f"{self.category}_{record_name}{filename_suffix}"
        )
        file_path = os.path.join(target_dir, base_name)
        
        content_parts = []
        for section_name, content in params:
            if content:
                content_parts.append(f"=== {section_name} ===\n{content}")
        
        if content_parts:
            combined_content = "\n\n\n".join(content_parts)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(combined_content)
    
    # ==================== 查询接口 ====================
    
    def get_step_count(self) -> int:
        """获取当前步骤计数（已记录的 log_record 次数）"""
        return len(self._step_records)
    
    def get_log_task_id(self) -> str:
        """获取日志文件名中使用的 task_id"""
        return self.task_id
    
    # ==================== 向后兼容接口 ====================
    # 以下方法保留旧 Trace 类的接口，便于现存代码无缝迁移
    
    @property
    def op_name(self) -> str:
        """兼容旧 Trace 的 op_name 属性（等价于 category）"""
        return self.category
    
    @property
    def trace_list(self) -> list:
        """兼容旧 Trace 的 trace_list 属性"""
        return self._step_records
    
    def save_parameters_to_files(self, agent_name: str, params: list):
        """兼容旧 Trace 的保存接口（内部调用 log_record 的文件写入逻辑，但不自增计数器）
        
        注意：此方法不自增步骤计数器，仅用于向后兼容。
        新代码应统一使用 log_record()。
        """
        if not self.log_dir:
            return
        
        expanded_log_dir = os.path.expanduser(self.log_dir)
        target_dir = os.path.join(expanded_log_dir, self.category)
        os.makedirs(target_dir, exist_ok=True)
        
        step = len(self._step_records)
        base_name = (
            f"Iteration{self.task_id}_Step{step:02d}_"
            f"{self.category}_{agent_name}_"
        )
        
        for param_name, content in params:
            file_path = os.path.join(target_dir, f"{base_name}{param_name}.txt")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(content))
    
    def insert_agent_record(self, agent_name: str, result: str = "",
                           prompt: str = "", reasoning: str = "",
                           error_log: str = "", profile_res: Optional[dict] = None,
                           session_id: Optional[str] = None,
                           elapsed_s: Optional[float] = None) -> None:
        """兼容旧 Trace 的 insert_agent_record 接口
        
        内部转换为 log_record() 调用。
        
        Args:
            agent_name: agent 名称
            result: 执行结果
            prompt: 使用的 prompt
            reasoning: 推理过程
            error_log: 错误日志（用于 verifier）
            profile_res: 性能数据（忽略）
            session_id: CLI session ID（忽略）
            elapsed_s: 耗时（忽略）
        """
        if agent_name in ("designer", "coder", "kernel_gen", "sketch",
                          "test_case_generator"):
            self.log_record(agent_name, [
                ('result', result),
                ('prompt', prompt),
                ('reasoning', reasoning),
            ])
        elif agent_name == "verifier":
            params = [('result', str(result))]
            if error_log:
                params.append(('error_log', error_log))
            self.log_record(agent_name, params)
        else:
            # 其他 agent 类型
            self.log_record(agent_name, [
                ('result', result),
                ('prompt', prompt),
                ('reasoning', reasoning),
            ])
    
    def insert_conductor_agent_record(self, res: str, prompt: str,
                                      reasoning: str,
                                      agent_name: str) -> None:
        """兼容旧 Trace 的 insert_conductor_agent_record 接口
        
        内部转换为 log_record() 调用，放入 conductor/ 子目录。
        """
        self.log_record(f"{agent_name}_decision", [
            ('result', res),
            ('prompt', prompt),
            ('reasoning', reasoning),
        ], subdirectory="conductor")
    
    def save_parsed_code(self, agent_name: str, params: list) -> None:
        """兼容旧 Trace 的 save_parsed_code 接口
        
        内部转换为 log_merged_record() 调用。
        """
        self.log_merged_record(agent_name, params)
