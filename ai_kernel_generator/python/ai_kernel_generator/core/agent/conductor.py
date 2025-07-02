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
from typing import List, Tuple
from ai_kernel_generator.core.trace import Trace
from ai_kernel_generator.core.utils import ActionType, ParsedCode
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory
from ai_kernel_generator.core.agent.agent_base import AgentBase

logger = logging.getLogger(__name__)


class Conductor(AgentBase):
    def __init__(self, op_name: str, task_id: str, log_dir: str, impl_type: str, model_config: dict) -> None:
        """
        初始化任务类，用于记录任务的各种属性。

        Args:
            op_name (str): 操作名称。
            task_id (str): 任务ID。
            log_dir (str): 日志目录。
            impl_type (str): 实现类型。
            model_config (dict): 模型配置。
        """

        self.op_name = op_name
        self.task_id = task_id
        self.log_dir = log_dir
        self.impl_type = impl_type
        self.model_config = model_config
        super().__init__(
            agent_name=f"Conductor -- [impl_type] {self.impl_type} -- [action] Check_Designer -- [op_name] {self.op_name}")

        self.step = 0
        self.fixed_step = 0
        self.trace = Trace(self.op_name, self.task_id, self.log_dir)
        self.code_parser = ParserFactory.get_code_parser()
        self.check_parser = ParserFactory.get_check_parser()
        self.check_format_instructions = self.check_parser.get_format_instructions()

        # 加载desginer检查模板
        self.check_designer_prompt = self.load_template("conductor/check_designer_template.j2")
        self.check_designer_base_doc = {}
        self.check_designer_input = {}

        # 根据impl_type选择不同的coder检查模板
        if self.impl_type == "triton":
            self.check_coder_prompt = self.load_template("conductor/check_triton_coder_template.j2")
        elif self.impl_type == "swft":
            self.check_coder_prompt = self.load_template("conductor/check_swft_coder_template.j2")
        else:
            raise ValueError(f"不支持的impl_type: {self.impl_type}")
        self.check_coder_base_doc = {}
        self.check_coder_input = {}

        # 加载错误分析模板
        self.analyze_error_prompt = self.load_template("conductor/analyze_error_template.j2")
        self.analyze_error_base_doc = {}
        self.analyze_error_input = {}

    def initialize_check_docs(self):
        """在Task初始化完成后调用，用trace.base_doc初始化check相关的文档"""
        if not self.trace.base_doc:
            logger.warning("trace.base_doc为空，无法初始化check_designer_base_doc")
            return

        self.check_designer_base_doc = {
            "op_name": self.trace.base_doc.get("op_name", ""),
            "task_desc": self.trace.base_doc.get("task_desc", ""),
            "aul_spec": self.load_doc("conductor_docs/check_aul.md"),
            "supported_compute_api": self.trace.base_doc.get("supported_compute_api", ""),
            "hardware_info": self.trace.base_doc.get("hardware_info", ""),
            "aul_tiling": self.trace.base_doc.get("aul_tiling", ""),
            "format_instructions": self.check_format_instructions,
        }
        self.check_designer_input = {
            "aul_code": "",
            **self.check_designer_base_doc,
        }
        logger.debug("check_designer_base_doc初始化完成")

        if self.impl_type == "triton":
            self.check_coder_base_doc = {
                "op_name": self.trace.base_doc.get("op_name", ""),
                "task_desc": self.trace.base_doc.get("task_desc", ""),
                "impl_type": self.impl_type,
                "triton_api_str": self.trace.base_doc.get("triton_api_str", ""),
                "triton_tutorial_str": self.trace.base_doc.get("triton_tutorial_str", ""),
                "triton_sample_code": self.trace.base_doc.get("triton_sample_code", ""),
                "format_instructions": self.check_format_instructions,
            }
            self.check_coder_input = {
                "aul_code": "",
                "triton_code": "",
                **self.check_coder_base_doc,
            }
        elif self.impl_type == "swft":
            self.check_coder_base_doc = {
                "op_name": self.trace.base_doc.get("op_name", ""),
                "task_desc": self.trace.base_doc.get("task_desc", ""),
                "impl_type": self.impl_type,
                "swft_sample_code": self.trace.base_doc.get("swft_sample_code", ""),
                "error_sample": self.trace.base_doc.get("error_sample", ""),
                "format_instructions": self.check_format_instructions,
            }
            self.check_coder_input = {
                "aul_code": "",
                "swft_code": "",
                "supported_api": "",
                **self.check_coder_base_doc,
            }
        logger.debug(f"check_coder_base_doc初始化完成，impl_type: {self.impl_type}")

        self.analyze_error_base_doc = {
            "impl_type": self.impl_type,
            "task_desc": self.trace.base_doc.get("task_desc", ""),
            "hardware_info": self.trace.base_doc.get("hardware_info", ""),
            "format_instructions": self.check_format_instructions,
        }
        self.analyze_error_input = {
            "designer_code": "",
            "coder_code": "",
            "error_log": "",
            "supported_api": "",
            **self.analyze_error_base_doc,
        }
        logger.debug("analyze_error_base_doc初始化完成")

    def find_last_parsed_code(self, action_types: List[ActionType]) -> str:
        """查找trace列表中最后出现的指定类型记录

        Args:
            action_types: 单个ActionType或ActionType列表，当为列表时查找最后出现的任一类型记录
        """
        parsed_code = ""
        for record in reversed(self.trace.trace_list):
            if record.action_type in action_types:
                parsed_result = ParserFactory.robust_parse(record.result, self.code_parser)
                parsed_code = parsed_result.code
                break
        return parsed_code

    async def get_next_action(self) -> Tuple[ActionType, ParsedCode, str]:
        """
        异步运行任务，执行操作任务字符串
        """
        try:
            self.step += 1
            action_type = ActionType.EXIT
            parsed_code = ParsedCode()
            suggestions = ""

            trace_len = len(self.trace.trace_list)
            pre_trace = self.trace.trace_list[trace_len - 1] if trace_len > 0 else None
            if pre_trace:
                action_type = pre_trace.action_type
                result = pre_trace.result
                if action_type in [ActionType.DO_DESIGNER, ActionType.FIX_DESIGNER,
                                   ActionType.DO_CODER, ActionType.FIX_CODER]:
                    logger.info(f"Task {self.task_id}, op_name: {self.op_name}, action_type: Conductor Self-Check")
                    return await self.self_check(action_type, result, parsed_code)
                elif action_type == ActionType.VERIFY:
                    if result == "False":
                        error_log = pre_trace.error_log
                        logger.info(
                            f"Task {self.task_id}, op_name: {self.op_name}, action_type: Conductor Analyze Error")
                        return await self.analyze_error(error_log, parsed_code)
                    else:
                        return ActionType.EXIT, parsed_code, suggestions
            return action_type, parsed_code, suggestions
        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")

    async def self_check(self, action_type: ActionType, result: str, parsed_code: ParsedCode):
        """一个简单的分析模块，解析code，并判断是否需要fix"""

        # 更新重试计数
        self._update_fixed_step(action_type)

        # 解析代码并设置到parsed_code
        code = self._parse_and_set_code(action_type, result, parsed_code)

        # 检查是否超过重试限制
        if self._should_force_next(action_type):
            logger.debug("修复次数超过限制，强制跳转")
            return self._get_force_next_action(action_type, parsed_code)

        # 检查代码是否为空
        if not code:
            logger.debug("代码为空，需要修复")
            return self._handle_empty_code(action_type, parsed_code)

        # 运行LLM检查
        return await self._run_llm_check(action_type, parsed_code)

    def _update_fixed_step(self, action_type: ActionType):
        """更新重试计数"""
        if action_type in [ActionType.DO_DESIGNER, ActionType.DO_CODER, ActionType.VERIFY]:
            self.fixed_step = 0
        elif action_type in [ActionType.FIX_DESIGNER, ActionType.FIX_CODER]:
            self.fixed_step += 1

    def _parse_and_set_code(self, action_type: ActionType, result: str, parsed_code: ParsedCode) -> str:
        """解析代码并设置到parsed_code对象"""
        parsed = ParserFactory.robust_parse(result, self.code_parser) if result else None
        code = parsed.code if parsed else ""

        if self._is_designer_action(action_type):
            parsed_code.aul_code = code
            self.check_designer_input["aul_code"] = code
            self.check_coder_input["aul_code"] = code
        elif self._is_coder_action(action_type):
            if self.impl_type == "triton":
                parsed_code.triton_code = code
                self.check_coder_input["triton_code"] = code
            elif self.impl_type == "swft":
                parsed_code.swft_code = code
                self.check_coder_input["swft_code"] = code
                self.check_coder_input["supported_api"] = self.trace.base_doc.get("supported_api", "")
        elif self._is_verifier_action(action_type):
            self.analyze_error_input["designer_code"] = parsed_code.aul_code
            if self.impl_type == "triton":
                self.analyze_error_input["coder_code"] = parsed_code.triton_code
            elif self.impl_type == "swft":
                self.analyze_error_input["coder_code"] = parsed_code.swft_code
                self.analyze_error_input["supported_api"] = self.trace.base_doc.get("supported_api", "")

        return code

    def _should_force_next(self, action_type: ActionType) -> bool:
        """检查是否应该强制跳转到下一步"""
        force_actions = [ActionType.FIX_DESIGNER, ActionType.FIX_CODER]
        return self.fixed_step > 1 and action_type in force_actions

    def _get_force_next_action(self, action_type: ActionType, parsed_code: ParsedCode):
        """获取强制跳转的下一步动作"""
        next_actions = {
            ActionType.FIX_DESIGNER: ActionType.DO_CODER,
            ActionType.FIX_CODER: ActionType.VERIFY,
            ActionType.VERIFY: ActionType.EXIT
        }
        next_action = next_actions.get(action_type, ActionType.EXIT)
        return next_action, parsed_code, ""

    def _handle_empty_code(self, action_type: ActionType, parsed_code: ParsedCode):
        """处理代码为空的情况"""
        if self._is_designer_action(action_type):
            logger.warning("Designer输出代码为空，需要修复")
            return ActionType.FIX_DESIGNER, parsed_code, ""
        elif self._is_coder_action(action_type):
            logger.warning("Coder输出代码为空，需要修复")
            return ActionType.FIX_CODER, parsed_code, ""

    async def _run_llm_check(self, action_type: ActionType, parsed_code: ParsedCode):
        """运行LLM检查"""
        if self._is_designer_action(action_type):
            self.agent_name = f"Conductor -- [impl_type] {self.impl_type} -- [action] Check_Designer -- [op_name] {self.op_name}"
            prompt = self.check_designer_prompt
            input = self.check_designer_input
        elif self._is_coder_action(action_type):
            self.agent_name = f"Conductor -- [impl_type] {self.impl_type} -- [action] Check_Coder -- [op_name] {self.op_name}"
            prompt = self.check_coder_prompt
            input = self.check_coder_input

        res, prompt, reasoning = await self.run_llm(
            prompt,
            input,
            self.model_config["conductor_check"]
        )
        self.trace.insert_conductor_record(res, prompt, reasoning, action_type)

        parsed = ParserFactory.robust_parse(res, self.check_parser)
        if not parsed:
            logger.warning("Conductor Self-Check 模块解析失败，默认推进流程")
            return self._get_next_action(action_type, parsed_code, "")

        result_correctness = parsed.result
        suggestions = parsed.suggestions if parsed.suggestions else ""

        if result_correctness == 1:
            logger.debug("Conductor Self-Check 模块决策：不需要修复")
            return self._get_next_action(action_type, parsed_code, "")
        elif result_correctness == 0:
            logger.debug("Conductor Self-Check 模块决策：需要修复")
            return self._get_fix_action(action_type, parsed_code, suggestions)
        else:
            logger.warning("Conductor Self-Check 模块决策结果异常，默认推进流程")
            return self._get_next_action(action_type, parsed_code, "")

    async def run_llm_analyze_error(self, parsed_code: ParsedCode) -> Tuple[str, str, str]:
        res, prompt, reasoning = await self.run_llm(
            self.analyze_error_prompt,
            self.analyze_error_input,
            self.model_config["conductor_analyze"]
        )
        self.trace.insert_conductor_record(res, prompt, reasoning, ActionType.VERIFY)

        parsed_result = ParserFactory.robust_parse(res, self.check_parser)
        if parsed_result.result == 1:
            action_type = ActionType.FIX_DESIGNER
        elif parsed_result.result == 2:
            action_type = ActionType.FIX_CODER

        suggestions = parsed_result.suggestions
        return action_type, parsed_code, suggestions

    def analyze_error(self, error_log: str, parsed_code: ParsedCode) -> Tuple[str, str, str]:
        designer_code = self.find_last_parsed_code([ActionType.DO_DESIGNER, ActionType.FIX_DESIGNER])
        coder_code = self.find_last_parsed_code([ActionType.DO_CODER, ActionType.FIX_CODER])
        parsed_code.aul_code = designer_code
        if self.impl_type == "triton":
            parsed_code.triton_code = coder_code
        elif self.impl_type == "swft":
            parsed_code.swft_code = coder_code

        # 更新重试计数
        self._update_fixed_step(ActionType.VERIFY)

        # 解析代码并设置到parsed_code
        self._parse_and_set_code(ActionType.VERIFY, "", parsed_code)
        self.analyze_error_input["error_log"] = error_log

        return self.run_llm_analyze_error(parsed_code)

    def _is_designer_action(self, action_type: ActionType) -> bool:
        """判断是否为Designer相关动作"""
        return action_type in [ActionType.DO_DESIGNER, ActionType.FIX_DESIGNER]

    def _is_coder_action(self, action_type: ActionType) -> bool:
        """判断是否为Coder相关动作"""
        return action_type in [ActionType.DO_CODER, ActionType.FIX_CODER]

    def _is_verifier_action(self, action_type: ActionType) -> bool:
        """判断是否为Verifier相关动作"""
        return action_type == ActionType.VERIFY

    def _get_next_action(self, action_type: ActionType, parsed_code: ParsedCode, suggestions: str):
        """获取下一步动作"""
        next_actions = {
            ActionType.DO_DESIGNER: ActionType.DO_CODER,
            ActionType.FIX_DESIGNER: ActionType.DO_CODER,
            ActionType.DO_CODER: ActionType.VERIFY,
            ActionType.FIX_CODER: ActionType.VERIFY,
            ActionType.VERIFY: ActionType.EXIT
        }
        next_action = next_actions.get(action_type, ActionType.EXIT)
        return next_action, parsed_code, suggestions

    def _get_fix_action(self, action_type: ActionType, parsed_code: ParsedCode, suggestions: str):
        """获取修复动作"""
        fix_actions = {
            ActionType.DO_DESIGNER: ActionType.FIX_DESIGNER,
            ActionType.FIX_DESIGNER: ActionType.FIX_DESIGNER,
            ActionType.DO_CODER: ActionType.FIX_CODER,
            ActionType.FIX_CODER: ActionType.FIX_CODER
        }
        fix_action = fix_actions.get(action_type, ActionType.EXIT)
        return fix_action, parsed_code, suggestions
