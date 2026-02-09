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
KernelDesigner Agent - 基于 Skill 系统的算法草图设计 Agent

负责：
- 根据用户输入和历史上下文生成算法草图（sketch）
- 使用 Skill 系统动态选择相关知识和策略
- 支持多种 DSL (Triton CUDA, Triton Ascend 等)
- 支持 Hint 模式（参数空间配置）
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from akg_agents import get_project_root

from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper
from akg_agents.core_v2.filesystem import ActionRecord
from akg_agents.utils.common_utils import ParserFactory, remove_copyright_from_text
from akg_agents.utils.hardware_utils import get_hardware_doc

# 导入 skill 系统模块
from akg_agents.core_v2.skill import (
    SkillLoader, 
    SkillSelector, 
    SelectionContext, 
    SkillLevel,
)

# 设置 Skills 目录路径
project_root = Path(get_project_root())
SKILLS_DIR = project_root / "op" / "resources" / "skills"

logger = logging.getLogger(__name__)


register_agent(scopes=["op"])
class KernelDesigner(AgentBase):
    """
    Kernel 算法草图设计 Agent
    
    基于 Skill 系统，根据任务需求动态选择知识和策略，生成高质量的算法草图。
    """
    
    # Agent 工具配置元数据
    TOOL_NAME = "call_kernel_designer"
    DESCRIPTION = """
仅生成算法设计方案/草图（sketch），不生成完整代码。

功能：
- 分析任务需求，设计算法实现方案
- 提供伪代码形式的算法草图
- 给出优化建议和实现策略
- 支持 Hint 模式（参数空间配置）

适用场景：
- 用户说只要"设计方案"、"算法草图"、"sketch"、"怎么实现"
- 只需要了解实现思路，不需要完整代码
- 想先讨论设计方案再决定是否生成代码
- 复杂算子需要先进行设计分析

⚠️ 注意：此工具仅生成设计方案，不生成可执行代码！
- 如果需要设计+代码生成，请使用 use_default_workflow 工具
- 设计方案用于指导后续代码生成

输出：算法草图（伪代码 + 优化策略 + 实现建议）
"""
    
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
                "description": "目标 DSL（例如：'triton_ascend', 'triton_cuda'）"
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
            "task_id": {
                "type": "string",
                "description": "任务 ID（可选）",
                "default": ""
            },
            "user_requirements": {
                "type": "string",
                "description": "用户额外需求（可选）",
                "default": ""
            },
            "enable_hint_mode": {
                "type": "boolean",
                "description": "是否启用 Hint 模式（参数空间配置）",
                "default": False
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
        "required": ["op_name", "task_desc", "dsl", "backend"]
    }
    
    def __init__(self, parser_config_path: str = None):
        """
        初始化 KernelDesigner Agent
        
        Args:
            parser_config_path: parser 配置文件路径（可选）
        """
        
        # 生成计数器（用于追踪生成步骤）
        self.design_step_count = 0
        self.parser_config_path = parser_config_path
        
        # 构建基础上下文
        context = {
            "agent_name": "kernel_designer",
            "task_label": "main",
        }
        
        # 初始化父类
        super().__init__(context=context)
        
        # ==================== Parser 初始化 ====================
        # 使用新的 parser loader
        from akg_agents.utils.parser_loader import create_agent_parser
        self.code_parser = create_agent_parser("designer", self.parser_config_path)
        if not self.code_parser:
            raise ValueError(
                "Failed to create kernel_designer parser. Please check your parser_config.yaml configuration.")
        self.format_instructions = self.code_parser.get_format_instructions()
        
        # ==================== Skill 系统初始化 ====================
        # 初始化 Skill Registry 和 Selector
        self._init_skills()
        
        # ==================== Prompt 模板初始化 ====================
        # 加载 jinja2 模板
        self.system_prompt_template = self.load_template("kernel_designer/system_prompt.j2")
        self.user_prompt_template = self.load_template("kernel_designer/user_prompt.j2")
    
    def _init_skills(self):
        """初始化 Skill 系统"""
        try:
            # 加载 skills（包含 designer 子目录）
            loader = SkillLoader()
            self.loaded_skills = loader.load_from_directory(SKILLS_DIR)
            
            logger.info(f"Loaded {len(self.loaded_skills)} skills from {SKILLS_DIR}")
            
            # 创建自定义过滤器
            # 1. Designer 专用过滤器：优先选择 designer 相关的 skills
            def designer_filter(skill, context):
                # 设计类 category 优先
                design_categories = ["design", "fundamental"]
                # 或者 metadata 中有 role: designer
                is_designer_skill = (
                    skill.category in design_categories or 
                    (skill.metadata and skill.metadata.get("role") == "designer")
                )
                # 排除纯实现类的 skills（由 coder 使用）
                implementation_only = skill.category == "implementation" and not is_designer_skill
                return not implementation_only
            
            # 创建 selector（带自定义过滤器）
            self.skill_selector = SkillSelector(
                custom_filters=[designer_filter]
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize skill system: {e}")
            # 设置为空以便降级处理
            self.loaded_skills = []
            self.skill_selector = None
    
    async def _select_skills(self, op_name: str = "", task_desc: str = "",
                             dsl: str = "", backend: str = "", 
                             enable_hint_mode: bool = False, has_hint: bool = False) -> List[Any]:
        """
        Designer 的 Skill 选择：
        1. 必选：sketch-design
        2. 可选：hint-mode（当 enable_hint_mode 且 has_hint）
        3. LLM 选择其他参考 Skills
        """
        if not self.skill_selector or not self.loaded_skills:
            return []
        
        try:
            skill_dict = {s.name: s for s in self.loaded_skills}
            selected_skills = []
            
            # 1. 必选：sketch-design
            if "sketch-design" in skill_dict:
                selected_skills.append(skill_dict["sketch-design"])
                logger.info("Selected skill: sketch-design (required)")
            
            # 2. hint-mode（可选）
            if enable_hint_mode and has_hint and "hint-mode" in skill_dict:
                selected_skills.append(skill_dict["hint-mode"])
                logger.info("Selected skill: hint-mode (hint mode enabled)")
            
            # 3. 粗筛可用的参考 Skills
            context = SelectionContext()
            filtered = self.skill_selector.coarse_filter(self.loaded_skills, context)
            
            # 排除已选
            already_selected = {s.name for s in selected_skills}
            candidates = [s for s in filtered if s.name not in already_selected and s.name != "hint-mode"]
            
            logger.info(f"Coarse filter: {len(self.loaded_skills)} -> {len(candidates)} candidates")
            
            if not candidates:
                return selected_skills
            
            # 4. LLM 选择参考 Skills
            skills_info = [{"name": s.name, "description": s.description} for s in candidates]
            
            llm_prompt = f"""你是一个 Skill 选择专家。请为以下算法草图设计任务选择相关的参考 Skills。

**任务信息**:
- 算子名称: {op_name}
- 目标 DSL: {dsl}
- 目标后端: {backend}
- 任务描述: {task_desc}

**可用参考 Skills**:
{json.dumps(skills_info, indent=2, ensure_ascii=False)}

**注意**：
1. 基于 Sketch 生成算子草图，无关的技能不要选择（如代码生成、测试、工作流等）
2. 选取最相关的2-3个 Skills 即可，不要选择太多

**输出格式**（JSON）:
 ```json
{{
"selected": ["skill-name-1", "skill-name-2", ...],
"reason": "简要选择理由"
}}
```
"""
            template = Jinja2TemplateWrapper("{{ prompt }}")
            response, _, _ = await self.run_llm(template, {"prompt": llm_prompt}, "standard")
            
            # 解析响应
            selected_names, reason = self._parse_llm_selection(response)
            for name in selected_names:
                if name in skill_dict and name not in already_selected:
                    selected_skills.append(skill_dict[name])
            
            logger.info(f"Designer selected {len(selected_skills)} skills: {[s.name for s in selected_skills]}")
            if reason:
                logger.info(f"Selection reason: {reason}")
            return selected_skills
            
        except Exception as e:
            logger.warning(f"Skill selection failed: {e}")
            return selected_skills if 'selected_skills' in locals() else []
    
    def _parse_llm_selection(self, response: str) -> Tuple[List[str], str]:
        """解析 LLM 返回的 skill 选择结果，返回 (名称列表, 理由)"""
        import re
        try:
            # 尝试解析 JSON 对象格式 {"selected": [...], "reason": "..."}
            json_match = re.search(r'\{[^{}]*"selected"[^{}]*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                return result.get("selected", []), result.get("reason", "")
            # 回退：尝试解析纯数组格式
            arr_match = re.search(r'\[.*?\]', response, re.DOTALL)
            if arr_match:
                return json.loads(arr_match.group()), ""
        except json.JSONDecodeError:
            pass
        return [], ""
    
    async def run(
        self,
        op_name: str,
        task_desc: str,
        dsl: str,
        backend: str,
        arch: str = "",
        task_id: str = "",
        user_requirements: str = "",
        enable_hint_mode: bool = False,
        history_compress: Optional[List[dict]] = None,
        model_level: str = "standard"
    ) -> Tuple[str, str, str]:
        """
        执行算法草图设计
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述
            dsl: 目标 DSL
            backend: 目标后端
            arch: 目标架构
            task_id: 任务 ID
            user_requirements: 用户额外需求
            enable_hint_mode: 是否启用 Hint 模式
            model_level: 模型级别
        
        Returns:
            Tuple[str, str, str]: (生成的草图, 完整 prompt, 推理过程)
        """
        try:
            # 确保 history_compress 不为 None
            if history_compress is None:
                history_compress = []
            
            # 检测是否有 hint（在 task_desc 中）
            has_hint = enable_hint_mode and ('hint' in task_desc.lower())
            if has_hint:
                logger.info(f"[Task {task_id}] 检测到 hint，启用 Hint 模式")
            
            # 1. 选择相关 Skills（异步），传递算子信息和 hint 模式
            selected_skills = await self._select_skills(
                op_name=op_name,
                task_desc=task_desc,
                dsl=dsl, 
                backend=backend, 
                enable_hint_mode=enable_hint_mode, 
                has_hint=has_hint
            )
            
            # 2. 渲染 System Prompt
            system_prompt = self.system_prompt_template.format(
                dsl=dsl,
                backend=backend,
                arch=arch
            )
            
            # 3. 获取硬件文档
            hardware_docs = get_hardware_doc(backend, arch)
            
            # 4. 渲染 User Prompt
            user_prompt = self.user_prompt_template.format(
                history_actions=history_compress,
                skills=selected_skills,
                op_name=op_name,
                task_desc=remove_copyright_from_text(task_desc),
                user_requirements=user_requirements,
                hardware_docs=hardware_docs,
                arch_name=arch,
                enable_hint_mode=enable_hint_mode,
                has_hint=has_hint,
                format_instructions=self.format_instructions
            )
            
            # 5. 组合完整 prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # 6. 创建 Jinja2 模板包装（用于 run_llm）
            template = Jinja2TemplateWrapper("{{ prompt }}")
            
            # 7. 更新上下文
            self.design_step_count += 1
            to_update_details = {
                "agent_name": "kernel_designer",
                "hash": task_id or "KernelDesigner" + "@" + str(self.design_step_count),
                "task_id": task_id,
                "step": self.design_step_count,
                "op_name": op_name,
                "dsl": dsl,
                "backend": backend,
                "arch": arch,
            }
            self.context.update(to_update_details)
            
            # 8. 调用 LLM
            llm_result, formatted_prompt, reasoning = await self.run_llm(
                template,
                {"prompt": full_prompt},
                model_level or "standard"
            )
            
            return llm_result, formatted_prompt, reasoning
        
        except Exception as e:
            logger.error(f"Exception in kernel_designer.run: {type(e).__name__}: {e}")
            raise
