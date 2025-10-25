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

import logging
from typing import Set, Optional, Dict, Any, List
from ai_kernel_generator.utils.workflow_manager import WorkflowManager
from ai_kernel_generator.utils.workflow_controller import WorkflowController
from ai_kernel_generator.utils.result_processor import ResultProcessor
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.core.trace import Trace
from ai_kernel_generator.utils.common_utils import ParserFactory

logger = logging.getLogger(__name__)


class Conductor(AgentBase):
    """
    Conductor，基于workflow.yaml配置文件进行工作流管理
    职责：工作流协调、状态管理
    """

    # 类级别的解析器缓存
    _conductor_parser = None

    @classmethod
    def get_conductor_parser(cls):
        """获取Conductor决策解析器"""
        if cls._conductor_parser is None:
            cls._conductor_parser = ParserFactory.get_conductor_parser()
        return cls._conductor_parser

    def __init__(self, op_name: str, task_desc: str, task_id: str, dsl: str,
                 framework: str, arch: str, workflow_config_path: Optional[str] = None,
                 config: Optional[dict] = None):
        """
        初始化Conductor

        Args:
            op_name: 算子名称
            task_desc: 任务描述
            task_id: 任务ID
            dsl: 实现类型
            framework: 前端框架
            arch: 硬件架构
            workflow_config_path: workflow配置文件路径，可选
            config: 完整配置字典，包含log_dir、model_config等
        """
        # 初始化基类
        context = {
            "agent_name": "conductor",
            "dsl": dsl,
            "op_name": op_name,
            "framework": framework,
            "arch": arch,
        }
        super().__init__(context=context, config=config)

        self.op_name = op_name
        self.task_desc = task_desc
        self.task_id = task_id
        self.dsl = dsl
        self.framework = framework
        self.arch = arch
        self.config = config

        # 从config中获取需要的属性
        if config:
            self.log_dir = config.get("log_dir")
            self.model_config = config.get("agent_model_config", {})
        else:
            raise ValueError("config is required for Conductor")

        self.step_count = 0
        self.llm_step_count = 0

        self.last_parse_success = True  # 记录最新agent执行的解析状态

        self.agent_parsers = {}

        self.conductor_template = self.load_template("conductor/analyze.j2")

        # 优先使用传入的workflow_config_path，否则从config中获取
        if workflow_config_path:
            self.workflow_config_path = workflow_config_path
        elif self.config and self.config.get('workflow_config_path'):
            self.workflow_config_path = self.config.get('workflow_config_path')
        else:
            raise ValueError("workflow_config_path is required for Conductor")
        self._load_workflow_config()

        self.trace = Trace(self.op_name, self.task_id, self.log_dir)

        self.task_info = {}

    def set_task_info(self, base_doc: Dict[str, Any] = None):
        """设置任务信息和基础文档"""
        # 初始化task_info
        self.task_info = WorkflowManager.initialize_task_info_fields(
            self.agent_info, self.op_name, self.task_id, self.dsl, self.task_desc, base_doc
        )

    def get_agent_history(self) -> List[str]:
        """从trace中获取agent执行历史"""
        return [record.agent_name for record in self.trace.trace_list]

    def _load_workflow_config(self):
        """从workflow.yaml文件读取配置信息到self属性中"""
        config = WorkflowManager.load_workflow_config(self.workflow_config_path)

        self.agent_info = config['agent_info']
        self.limitation_info = config['limitation_info']
        self.start_agent = config['start_agent']
        self.max_step = config['max_step']
        self.repeat_limits = config['repeat_limits']
        self.agent_next_mapping = config['agent_next_mapping']
        self.mandatory_llm_analysis = config['mandatory_llm_analysis']

    def get_agent_parser(self, agent_name: str):
        """为指定的agent获取解析器"""
        return ResultProcessor.get_agent_parser(agent_name, self.workflow_config_path, self.agent_parsers)

    def record_agent_execution(self, agent_name: str, result: str, prompt: str = "", reasoning: str = "",
                               error_log: str = "", profile_res: dict = None) -> bool:
        """
        记录agent执行结果，进行解析并更新任务信息

        Args:
            agent_name: agent名称（designer, coder, verifier，...）
            result: 执行结果（json字符串）
            prompt: 使用的prompt
            reasoning: 推理过程
            error_log: 错误日志（主要用于verifier）
            profile_res: 性能分析结果字典（主要用于verifier），包含：
                - gen_time: 生成代码执行时间（微秒）
                - base_time: 基准代码执行时间（微秒）
                - speedup: 加速比
                - autotune_summary: autotune配置详情（可选，仅triton+ascend）

        Returns:
            bool: 解析是否成功（对于不需要解析器的agent返回True）
        """
        parse_success = True

        try:
            # 1. 保存原始数据到trace
            self.trace.insert_agent_record(
                agent_name=agent_name,
                result=result,
                prompt=prompt,
                reasoning=reasoning,
                error_log=error_log,
                profile_res=profile_res
            )

            # 2. 进行解析并更新任务信息
            agent_config = self.agent_info.get(agent_name, {})
            if 'output_format' in agent_config:
                # 需要解析器的agent
                agent_parser = self.get_agent_parser(agent_name)
                parse_success = ResultProcessor.parse_and_update_code(
                    agent_name, result, self.task_info, agent_parser, self.trace, self.agent_info
                )
            elif agent_name == "verifier":
                ResultProcessor.update_verifier_result(result, error_log, self.task_info, profile_res)

        except Exception as e:
            logger.error(f"Failed to record and process agent execution for {agent_name}: {e}")
            parse_success = False

        # 记录最新的解析状态
        self.last_parse_success = parse_success
        return parse_success

    def get_illegal_agent(self) -> Set[str]:
        """获取违禁操作的agent集合"""
        return WorkflowController.get_illegal_agent(
            self.step_count, self.max_step, self.get_current_agent_name(),
            self.get_agent_history(), self.repeat_limits, self.agent_info
        )

    def get_valid_next_agent(self, agent_name: str) -> Set[str]:
        """获取有效的下一个agent选项"""
        return WorkflowController.get_valid_next_agent(
            agent_name, self.agent_next_mapping, self.step_count, self.max_step,
            self.get_current_agent_name(), self.get_agent_history(), self.repeat_limits, self.agent_info
        )

    def get_current_agent_name(self) -> Optional[str]:
        """获取当前agent名称，从trace中获取最新的agent"""
        if self.trace.trace_list:
            return self.trace.trace_list[-1].agent_name
        return None

    def _should_retry_current_agent(self, current_agent: str) -> bool:
        """
        检查当前agent是否需要重试（基于解析失败）

        Args:
            current_agent: 当前agent名称

        Returns:
            是否需要重试
        """
        if not current_agent:
            return False

        # 检查是否可以重试（当前agent在valid_next_agents中）
        valid_agents = self.get_valid_next_agent(current_agent)
        if current_agent not in valid_agents:
            return False

        # 检查解析是否失败
        if not self.last_parse_success:
            logger.warning(f"Agent {current_agent} latest execution failed to parse, will retry")
            return True

        return False

    async def _llm_decide_next_agent(self, current_agent: str, valid_next_agents: Set[str]) -> str:
        """
        使用LLM决策下一个agent，基于analyze.j2模板

        Args:
            current_agent: 当前agent名称
            valid_next_agents: 有效的下一个agent集合

        Returns:
            str: 决策的下一个agent名称
        """
        try:
            # 获取Conductor解析器和格式说明
            conductor_parser = self.get_conductor_parser()
            format_instructions = conductor_parser.get_format_instructions()

            # 获取最新的coder代码结果
            agent_result = self.task_info.get('coder_code', '')

            # 获取错误日志（如果有）
            error_log = self.task_info.get('verifier_error', '')

            # 构建输入数据（匹配analyze.j2模板）
            input_data = {
                'dsl': self.dsl,
                'expert_suggestion': self.task_info.get('expert_suggestion', ''),
                'op_name': self.op_name,
                'framework': self.framework,
                'task_desc': self.task_desc,
                'agent_name': current_agent,
                'agent_result': agent_result,
                'error_log': error_log[:5000] if error_log else None,
                'valid_next_agents': ', '.join(sorted(valid_next_agents)),
                'format_instructions': format_instructions,
            }

            # 执行LLM生成前更新context，确保正确性
            self.llm_step_count += 1
            to_update_context = {
                "agent_name": "conductor",
                "hash": self.task_id + "@" + str(self.llm_step_count),
                "task_id": self.task_id,
                "backend": self.task_info.get("backend", ""),
                "task_desc": self.task_desc,
                "step": self.llm_step_count,
                "workflow_name": self.task_info.get("workflow_name", ""),
            }
            self.context.update(to_update_context)

            model_name = self.model_config.get('conductor')
            content, formatted_prompt, reasoning = await self.run_llm(self.conductor_template, input_data, model_name)

            # 保存LLM调用记录
            self.trace.insert_conductor_agent_record(
                res=content,
                prompt=formatted_prompt,
                reasoning=reasoning,
                agent_name="decision"
            )

            # 解析LLM输出
            agent_decision, suggestion = ResultProcessor.parse_conductor_decision(
                content, conductor_parser, valid_next_agents
            )

            if agent_decision:
                # 保存suggestion到task_info用于传递给下一个agent
                if suggestion:
                    self.task_info['conductor_suggestion'] = suggestion
                return agent_decision

        except Exception as e:
            logger.warning(f"LLM decision failed: {e}, falling back to default")

        logger.warning(f"LLM decision failed for agent {current_agent}, ending task with finish")
        return "finish"

    async def get_next_agent(self) -> str:
        """
        获取下一个要执行的agent
        自动从trace中提取最新的agent并处理其结果，再决策下一个agent
        """
        # 这个方法只应该在执行过agent后调用，用于决策下一个agent
        if not self.trace.trace_list:
            raise ValueError("Error: get_next_agent() should only be called after agent execution.")

        self.step_count += 1

        # 清除上一次的conductor建议，避免影响新的决策
        self.task_info.pop('conductor_suggestion', None)

        current_agent = self.get_current_agent_name()

        # 检查当前agent是否解析失败，如果失败且可以重试，就返回同一个agent
        if self._should_retry_current_agent(current_agent):
            logger.info(f"Agent {current_agent} parsing failed, retrying...")
            return current_agent

        # 根据yaml文件要求，获取next_agent_name
        valid_next_agents = self.get_valid_next_agent(current_agent)

        # 特殊处理verifier的结果
        if current_agent == "verifier":
            verifier_result = self.task_info.get('verifier_result', False)
            if verifier_result:
                return "finish"
            else:
                # 验证失败，排除finish选项
                valid_next_agents.discard("finish")

        # 根据valid_next_agents数量和mandatory_llm_analysis配置决定是否需要LLM分析
        if len(valid_next_agents) == 0:
            # 没有可选agent，直接结束
            return "finish"
        elif len(valid_next_agents) == 1:
            # 只有一个可选agent，根据mandatory_llm_analysis判断是否需要LLM分析
            next_agent = list(valid_next_agents)[0]
            if current_agent in self.mandatory_llm_analysis:
                # 需要强制LLM分析
                logger.info(f"Agent {next_agent} requires mandatory LLM analysis")
                decided_agent = await self._llm_decide_next_agent(current_agent, valid_next_agents)
                return decided_agent
            else:
                # 直接返回该agent，无需LLM分析
                logger.info(f"Direct transition to {next_agent} (no LLM analysis required)")
                return next_agent
        else:
            # 多个可选agent，调用LLM进行决策
            logger.info(f"Multiple valid agents {valid_next_agents}, using LLM decision")
            next_agent = await self._llm_decide_next_agent(current_agent, valid_next_agents)
            return next_agent
