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
Selector Agent - 智能筛选相关的手写优化建议
"""

import logging
from typing import List, Dict, Any
from ai_kernel_generator.core.agent.agent_base import AgentBase
from ai_kernel_generator.utils.common_utils import ParserFactory

logger = logging.getLogger(__name__)


class Selector(AgentBase):
    """
    Selector Agent - 根据任务特征筛选相关的手写优化建议
    
    功能：
    1. 接收任务描述和所有候选文档的完整内容
    2. 使用LLM分析任务特征和文档内容
    3. 筛选出所有相关的优化建议文档
    """
    
    def __init__(
        self,
        op_name: str,
        task_desc: str,
        dsl: str = "triton",
        config: dict = None,
    ):
        """初始化Selector Agent
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述
            dsl: DSL类型
            config: 配置字典
        """
        self.op_name = op_name
        self.task_desc = task_desc
        self.dsl = dsl
        self.llm_step_count = 0
        
        if not config:
            raise ValueError("config is required for Selector")
        
        self.model_config = config.get("agent_model_config", {})
        
        context = {
            "agent_name": "selector",
            "dsl": self.dsl,
            "op_name": self.op_name,
        }
        super().__init__(context=context, config=config)
        
        # 创建解析器
        self.selector_parser = ParserFactory.get_selector_parser()
        self.format_instructions = self.selector_parser.get_format_instructions()
        
        # 加载模板
        self.selector_prompt = self.load_template("handwrite/select_relevant.j2")
        
        logger.debug(f"Selector Agent initialized for {op_name}")
    
    async def run(self, candidates: List[Dict[str, str]]) -> List[str]:
        """执行筛选，返回相关文档的名称列表
        
        Args:
            candidates: 候选文档列表，每个元素包含：
                - name: 文档名称
                - torch_code: torch代码
                - triton_code: triton代码
                - improvement: 优化建议
        
        Returns:
            List[str]: 相关文档的名称列表
        """
        if not candidates:
            logger.warning("No candidates provided for selection")
            return []
        
        logger.info(f"Selecting relevant documents from {len(candidates)} candidates for {self.op_name}")
        
        # 构建输入数据
        input_data = {
            "op_name": self.op_name,
            "task_desc": self.task_desc,
            "candidates": candidates,
            "total_count": len(candidates),
            "format_instructions": self.format_instructions,
        }
        
        # 更新context
        self.llm_step_count += 1
        to_update_context = {
            "agent_name": "selector",
            "hash": f"selector_{self.op_name}",
            "task_id": "",
            "step": self.llm_step_count,
        }
        self.context.update(to_update_context)
        
        # 获取模型配置
        model_config = self.model_config.get("selector") or self.model_config.get("default")
        if not model_config:
            logger.warning("No model config found for selector, returning all candidates")
            return [c['name'] for c in candidates]
        
        # 执行LLM筛选
        try:
            content, _, _ = await self.run_llm(
                self.selector_prompt, input_data, model_config
            )
            
            # 解析结果
            try:
                parsed_result = ParserFactory.robust_parse(content, self.selector_parser)
                selected_names = parsed_result.selected_names
                
                # 验证选择的文档是否存在
                available_names = {c['name'] for c in candidates}
                valid_names = []
                invalid_names = []
                
                for name in selected_names:
                    if name in available_names:
                        valid_names.append(name)
                    else:
                        invalid_names.append(name)
                
                # 报告无效的文档名称
                if invalid_names:
                    logger.warning(f"LLM returned {len(invalid_names)} invalid document names (spelling errors or non-existent): {invalid_names}")
                
                if valid_names:
                    logger.info(f"Selected {len(valid_names)}/{len(candidates)} relevant documents: {valid_names}")
                    return valid_names
                else:
                    logger.warning("No valid documents selected, returning all candidates")
                    return [c['name'] for c in candidates]
                    
            except Exception as parse_error:
                logger.error(f"Failed to parse selector result: {parse_error}, returning all candidates")
                return [c['name'] for c in candidates]
                
        except Exception as e:
            logger.error(f"Failed to run selector LLM: {e}, returning all candidates")
            return [c['name'] for c in candidates]

