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
KernelGen Agent - 基于 Skill 系统的内核代码生成 Agent

负责：
- 根据用户输入和历史上下文生成内核代码
- 使用 Skill 系统动态选择相关知识和策略
- 支持多种 DSL (Triton CUDA, Triton Ascend 等)
- 支持多种框架 (PyTorch, MindSpore 等)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from akg_agents import get_project_root

from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper
from akg_agents.core_v2.filesystem import ActionRecord

# 导入 skill 系统模块
from akg_agents.core_v2.skill import (
    SkillLoader, 
    SkillSelector, 
    SelectionContext, 
    SkillLevel,
    create_metadata_matcher
)

# 设置 Skills 目录路径
project_root = Path(get_project_root())
SKILLS_DIR = project_root.parent.parent / "examples" / "run_skill" / "skills"

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class KernelGen(AgentBase):
    """
    Kernel 代码生成 Agent
    
    基于 Skill 系统，根据任务需求动态选择知识和策略，生成高性能内核代码。
    """
    
    # Agent 工具配置元数据
    TOOL_NAME = "call_kernel_gen"
    DESCRIPTION = "根据任务需求生成高性能内核代码。"
    
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {
                "type": "string",
                "description": "算子名称"
            },
            "task_desc": {
                "type": "string",
                "description": "任务描述或算法规格说明"
            },
            "dsl": {
                "type": "string",
                "description": "目标 DSL（例如：'triton-ascend', 'triton-cuda'）"
            },
            "framework": {
                "type": "string",
                "description": "目标框架（例如：'torch', 'mindspore', 'numpy'）"
            },
            "backend": {
                "type": "string",
                "description": "目标硬件后端（例如：'cuda', 'ascend'）"
            },
            "arch": {
                "type": "string",
                "description": "目标硬件架构（例如：'a100', 'ascend910b4'）",
                "default": ""
            },
            "user_requirements": {
                "type": "string",
                "description": "用户额外需求（可选）",
                "default": ""
            },
            "task_id": {
                "type": "string",
                "description": "任务 ID（可选）",
                "default": ""
            },
            "history_compress": {
                "type": "array",
                "description": "压缩后的历史记录列表（可选）",
                "items": {
                    "type": "object"
                },
                "default": []
            },
            "model_level": {
                "type": "string",
                "description": "模型级别（例如：'standard', 'fast', 'complex'）",
                "default": "standard"
            }
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend"]
    }
    
    def __init__(self, parser_config_path: str = None):
        """
        初始化 KernelGen Agent
        
        Args:
            parser_config_path: parser 配置文件路径（可选）
        """
        
        # 生成计数器（用于追踪生成步骤）
        self.codegen_step_count = 0
        self.parser_config_path = parser_config_path
        
        # 构建基础上下文
        context = {
            "agent_name": "kernel_gen",
            "task_label": "main",
        }
        
        # 初始化父类
        super().__init__(context=context)
        
        # ==================== Parser 初始化 ====================
        # 使用新的 parser loader
        from akg_agents.utils.parser_loader import create_agent_parser
        self.code_parser = create_agent_parser("kernel_gen", self.parser_config_path)
        if not self.code_parser:
            raise ValueError(
                "Failed to create kernel_gen parser. Please check your parser_config.yaml configuration.")
        self.format_instructions = self.code_parser.get_format_instructions()
        
        # ==================== Skill 系统初始化 ====================
        # 初始化 Skill Registry 和 Selector
        self._init_skills()
        
        # ==================== Prompt 模板初始化 ====================
        # 加载 jinja2 模板
        self.system_prompt_template = self.load_template("kernel_gen/system_prompt.j2")
        self.user_prompt_template = self.load_template("kernel_gen/user_prompt.j2")
    
    def _init_skills(self):
        """初始化 Skill 系统"""
        try:
            # 加载 skills
            loader = SkillLoader()
            self.loaded_skills = loader.load_from_directory(SKILLS_DIR)
            
            logger.info(f"Loaded {len(self.loaded_skills)} skills from {SKILLS_DIR}")
            
            # 创建自定义过滤器
            # 1. 排除无关 category
            def category_filter(skill, context):
                unrelated = ["writing", "web", "communication", "documentation", "workflow"]
                return skill.category not in unrelated
            
            # 2. backend 匹配器
            backend_filter = create_metadata_matcher("backend")
            
            # 3. dsl 匹配器
            dsl_filter = create_metadata_matcher("dsl")
            
            # 4. framework 匹配器
            framework_filter = create_metadata_matcher("framework")
            
            # 创建 selector（带自定义过滤器）
            self.skill_selector = SkillSelector(
                custom_filters=[category_filter, backend_filter, dsl_filter, framework_filter]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize skill system: {e}")
            # 设置为空以便降级处理
            self.loaded_skills = []
            self.skill_selector = None
    
    
    async def _select_skills(self, dsl: str = "", framework: str = "", backend: str = "") -> List[Any]:
        """
        选择相关的 Skills
        
        使用 SkillSelector 的自定义过滤器进行两阶段筛选：
        1. 粗筛（coarse_filter）：使用 custom_filters 自动过滤
        2. 精筛（LLM）：根据任务上下文智能选择
        
        Args:
            dsl: 目标 DSL
            framework: 目标框架
            backend: 目标后端
        
        Returns:
            选中的 Skill 列表
        """
        if not self.skill_selector:
            logger.warning("Skill system not initialized, skipping skill selection")
            return []
        
        if not self.loaded_skills:
            logger.warning("No skills loaded, skipping skill selection")
            return []
        
        # 构建选择上下文（包含 dsl, framework, backend 供过滤器使用）
        context = SelectionContext(
            custom_fields={
                "task_type": "code_generation",
                "framework": framework or "unknown",
                "optimization_goal": f"生成 {dsl or 'kernel'} 内核代码",
                "dsl": dsl,
                "framework": framework,
                "backend": backend
            }
        )
        
        try:
            # 阶段1：粗筛（使用 custom_filters 自动过滤）
            candidates = self.skill_selector.coarse_filter(self.loaded_skills, context)
            logger.info(f"Coarse filter: {len(self.loaded_skills)} -> {len(candidates)} skills")
            
            # 阶段2：LLM 精筛
            # 定义 prompt 模板
            prompt_template = """你是一个 Skill 选择专家。现在需要为内核代码生成（kernel code generation）任务选择相关的 Skills。

**任务上下文**:
{context_str}

**候选 Skills（已粗筛）**:
{skills_str}

**任务要求**:
请从候选 Skills 中选择与内核代码生成任务最相关的 Skills。注意：
1. 优先选择与 DSL 和框架直接相关的 Skills
2. 包含代码生成方法和实现技巧相关的 Skills
3. 确保不要遗漏重要的 Skills
4. 重点是内核代码生成，无关的技能不要选择（如算子草图设计、测试等）

**输出格式**（JSON）:
```json
{{
  "selected": ["skill-name-1", "skill-name-2", ...],
  "reason": "选择理由"
}}
```
"""
            
            prompt = self.skill_selector.build_llm_prompt(candidates, context, prompt_template)
            
            template = Jinja2TemplateWrapper("{{ prompt }}")
            llm_response, _, _ = await self.run_llm(
                template, 
                {"prompt": prompt}, 
                "standard"
            )
            
            selected_skills = self.skill_selector.parse_llm_response(llm_response, candidates)
            
            logger.info(f"✓ Selected {len(selected_skills)} skills: {[s.name for s in selected_skills]}")
            
            return selected_skills
        
        except Exception as e:
            logger.warning(f"Skill selection failed: {e}, returning empty list")
            return []
        
    
    async def run(
        self,
        op_name: str,
        task_desc: str,
        dsl: str,
        framework: str,
        backend: str,
        arch: str = "",
        user_requirements: str = "",
        task_id: str = "",
        history_compress: Optional[List[dict]] = None,
        model_level: str = "standard"
    ) -> Tuple[str, str, str]:
        """
        执行代码生成
        
        Returns:
            Tuple[str, str, str]: (生成的代码, 完整 prompt, 推理过程)
        """
        try:
            # 确保 history_compress 不为 None
            if history_compress is None:
                history_compress = []
            
            # 生成函数名
            func_name = f"{op_name}_{dsl}_{framework}"
            
            # 1. 选择相关 Skills（异步）
            selected_skills = await self._select_skills(dsl=dsl, framework=framework, backend=backend)
            
            # 2. 渲染 System Prompt
            system_prompt = self.system_prompt_template.format(
                dsl=dsl,
                framework=framework,
                backend=backend,
                arch=arch
            )
            
            # 3. 渲染 User Prompt
            user_prompt = self.user_prompt_template.format(
                history_actions=history_compress,
                skills=selected_skills,
                op_name=op_name,
                func_name=func_name,
                task_desc=task_desc,
                user_requirements=user_requirements,
                format_instructions=self.format_instructions
            )
            
            # 4. 组合完整 prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # 5. 创建 Jinja2 模板包装（用于 run_llm）
            template = Jinja2TemplateWrapper("{{ prompt }}")
            
            # 6. 更新上下文
            self.codegen_step_count += 1
            to_update_details = {
                "agent_name": "kernel_gen",
                "hash": task_id or "KernelGen" + "@" + str(self.codegen_step_count),
                "task_id": task_id,
                "step": self.codegen_step_count,
                "op_name": op_name,
                "dsl": dsl,
                "framework": framework,
                "backend": backend,
                "arch": arch,
            }
            self.context.update(to_update_details)
            
            # 7. 调用 LLM
            generated_code, formatted_prompt, reasoning = await self.run_llm(
                template,
                {"prompt": full_prompt},
                model_level or "standard"
            )
            
            return generated_code, formatted_prompt, reasoning
        
        except Exception as e:
            logger.error(f"Exception in kernel_gen.run: {type(e).__name__}: {e}")
            raise
