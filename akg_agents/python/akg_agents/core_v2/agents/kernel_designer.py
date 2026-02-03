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
- 支持进化优化（inspirations）
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
    SkillLevel
)

# 设置 Skills 目录路径
project_root = Path(get_project_root())
SKILLS_DIR = project_root.parent.parent / "examples" / "run_skill" / "skills"

logger = logging.getLogger(__name__)


def format_inspirations(inspirations: List[dict]) -> str:
    """
    将 inspirations 列表转换为格式化字符串
    
    Args:
        inspirations: 包含字典的列表，每个字典格式为:
            {
                'strategy_mode': str,
                'impl_code': str,
                'sketch': str,
                'profile': {
                    'gen_time': float,
                    'base_time': float,
                    'speedup': float,
                    'autotune_summary': str (可选)
                },
                'is_parent': bool
            }
    
    Returns:
        str: 拼接后的字符串，包含所有 sketch/impl_code 和 profile 信息
    """
    if not inspirations:
        return ""
    
    result_parts = []
    has_parent = False
    
    for i, inspiration in enumerate(inspirations):
        if not isinstance(inspiration, dict):
            logger.warning(f"跳过非字典类型的 inspiration: {type(inspiration)}")
            continue
        
        sketch = inspiration.get('sketch', '')
        impl_code = inspiration.get('impl_code', '')
        profile = inspiration.get('profile', {})
        is_parent = inspiration.get('is_parent', False)
        
        # 检测是否有父代
        if is_parent:
            has_parent = True
        
        if sketch or impl_code:  # 只有当 sketch 或 impl_code 不为空时才添加
            # 处理 profile 信息（dict 格式）
            gen_time = profile.get('gen_time', float('inf'))
            base_time = profile.get('base_time', 0.0)
            speedup = profile.get('speedup', 0.0)
            autotune_summary = profile.get('autotune_summary', '')
            
            if gen_time != float('inf'):
                profile_text = f"根据此方案草图生成的代码计算耗时: {gen_time:.4f}us, 基准代码耗时: {base_time:.4f}us, 加速比: {speedup:.2f}x"
                # 如果有 autotune 信息，添加到 profile_text
                if autotune_summary:
                    profile_text += f"\n\nAutotune配置详情:\n{autotune_summary}"
            else:
                profile_text = "代码执行耗时: N/A"
            
            # 如果是父代，添加标记
            parent_mark = " 【父代方案】" if is_parent else ""
            inspiration_text = f"## Inspiration {i+1}{parent_mark} {profile_text}\n"
            if sketch:
                inspiration_text += f"算法草图：\n```\n{sketch}\n```\n"
            if impl_code:
                inspiration_text += f"代码：\n```\n{impl_code}\n```\n"
            result_parts.append(inspiration_text)
    
    # 如果有父代，在开头添加进化优化策略说明
    if has_parent and result_parts:
        strategy_note = (
            "**进化优化策略**：\n"
            "- 标记为【父代方案】的是本次进化的基础，请以它为主要参考进行改进和优化\n"
            "- 其他 Inspiration 可作为补充参考，用于交叉变异和借鉴优化思路\n"
            "- 请在父代方案的基础上，结合其他方案的优点，生成优化后的草图\n\n"
        )
        result_parts.insert(0, strategy_note)
    
    return "\n".join(result_parts)


@register_agent(scopes=["op"])
class KernelDesigner(AgentBase):
    """
    Kernel 算法草图设计 Agent
    
    基于 Skill 系统，根据任务需求动态选择知识和策略，生成高质量的算法草图。
    """
    
    # Agent 工具配置元数据
    TOOL_NAME = "call_kernel_designer"
    DESCRIPTION = "根据任务需求生成算法草图（sketch）。"
    
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
            "inspirations": {
                "type": "array",
                "description": "进化优化的参考方案列表（可选）",
                "items": {
                    "type": "object"
                },
                "default": []
            },
            "conductor_suggestion": {
                "type": "string",
                "description": "Conductor 的建议（可选）",
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
        
        # ==================== 加载 Sketch 设计指南 ====================
        # 这是保证输出 sketch DSL 格式的关键文档（348行的详细规范）
        self.sketch_guide = self._load_sketch_guide()
    
    def _load_sketch_guide(self) -> str:
        """
        加载 sketch 设计指南文档（SKETCH_DESIGN_v2.md）
        
        这是原版 Designer 的核心文档，定义了 sketch DSL 的语法规范。
        不加载此文档会导致 LLM 生成 Markdown 文档而不是 sketch 代码。
        
        Returns:
            str: 文档内容，如果加载失败则返回空字符串
        """
        try:
            sketch_guide_path = Path(get_project_root()) / "op" / "resources" / "docs" / "sketch_docs" / "SKETCH_DESIGN_v2.md"
            
            if sketch_guide_path.exists():
                with open(sketch_guide_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.info(f"✓ Loaded sketch guide ({len(content)} chars)")
                    return content
        except Exception as e:
            logger.warning(f"Failed to load sketch guide: {e}")
        
        logger.warning("⚠️  Sketch design guide not found, output format may be incorrect")
        return ""
    
    def _init_skills(self):
        """初始化 Skill 系统"""
        try:
            # 加载 skills
            loader = SkillLoader()
            self.loaded_skills = loader.load_from_directory(SKILLS_DIR)
            
            logger.info(f"Loaded {len(self.loaded_skills)} skills from {SKILLS_DIR}")
            
            # 创建 selector
            self.skill_selector = SkillSelector()
            
        except Exception as e:
            logger.error(f"Failed to initialize skill system: {e}")
            # 设置为空以便降级处理
            self.loaded_skills = []
            self.skill_selector = None
    
    async def _select_skills(self, dsl: str = "", backend: str = "") -> List[Any]:
        """
        选择相关的 Skills
        
        1. 排除无关的 skill（基于 category）
        2. 使用 metadata.get() 进行初筛
        3. 使用 LLM 进行精筛
        
        Args:
            dsl: 目标 DSL
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
        
        # 阶段1：使用 metadata 和 category 进行初筛
        filtered_skills = []
        for skill in self.loaded_skills:
            # 1. 排除无关的 skill（基于 category）
            unrelated_categories = ["writing", "web", "communication", "documentation", "workflow"]
            if skill.category in unrelated_categories:
                logger.debug(f"Filtered out unrelated skill ({skill.category}): {skill.name}")
                continue
            
            # 2. 如果 Skill 有 backend/dsl metadata，检查是否匹配
            # 如果没有这些 metadata，则保留（通用 Skill）
            skill_backend = skill.metadata.get("backend")
            skill_dsl = skill.metadata.get("dsl")
            
            # 如果 Skill 指定了 backend，但不匹配，则排除
            if skill_backend and backend and skill_backend != backend:
                logger.debug(f"Filtered out skill {skill.name}: backend mismatch ({skill_backend} != {backend})")
                continue
            
            # 如果 Skill 指定了 dsl，但不匹配，则排除
            if skill_dsl and dsl and skill_dsl != dsl:
                logger.debug(f"Filtered out skill {skill.name}: dsl mismatch ({skill_dsl} != {dsl})")
                continue
            
            filtered_skills.append(skill)
        
        logger.info(f"Initial filter: {len(self.loaded_skills)} -> {len(filtered_skills)} skills (removed workflow and unrelated categories)")
        
        # 阶段2：使用 LLM 进行精筛
        # 构建选择上下文
        context = SelectionContext(
            task_type="sketch_design",
            framework=backend or "unknown",
            optimization_goal=f"设计 {dsl or backend or 'kernel'} 算子草图，专注于算法层面的优化策略"
        )
        
        try:
            # 1. 构建 LLM prompt
            prompt = self.skill_selector.build_llm_prompt(filtered_skills, context)
            
            # 2. 调用 LLM（异步）
            template = Jinja2TemplateWrapper("{{ prompt }}")
            llm_response, _, _ = await self.run_llm(
                template, 
                {"prompt": prompt}, 
                "standard"
            )
            
            # 3. 解析 LLM 响应
            selected_skills = self.skill_selector.parse_llm_response(llm_response, filtered_skills)
            
            logger.info(f"✓ Selected {len(selected_skills)} skills: {[s.name for s in selected_skills]}")
            import pdb; pdb.set_trace()
            return selected_skills
        
        except Exception as e:
            logger.warning(f"Skill selection failed: {e}, returning empty list")
            return []
    
    async def run(
        self,
        op_name: str,
        task_desc: str,
        dsl: str,
        backend: str,
        arch: str = "",
        task_id: str = "",
        inspirations: Optional[List[dict]] = None,
        conductor_suggestion: str = "",
        user_requirements: str = "",
        enable_hint_mode: bool = False,
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
            inspirations: 进化优化的参考方案列表
            conductor_suggestion: Conductor 建议
            user_requirements: 用户额外需求
            enable_hint_mode: 是否启用 Hint 模式
            model_level: 模型级别
        
        Returns:
            Tuple[str, str, str]: (生成的草图, 完整 prompt, 推理过程)
        """
        try:
            # 确保参数不为 None
            if inspirations is None:
                inspirations = []
            
            # 检测是否有 hint（在 task_desc 中）
            has_hint = enable_hint_mode and ('hint' in task_desc.lower())
            if has_hint:
                logger.info(f"[Task {task_id}] 检测到 hint，启用 Hint 模式")
            
            # 1. 选择相关 Skills（异步）
            selected_skills = await self._select_skills(dsl=dsl, backend=backend)
            
            # 2. 渲染 System Prompt
            system_prompt = self.system_prompt_template.format(
                dsl=dsl,
                backend=backend,
                arch=arch
            )
            
            # 3. 格式化 inspirations
            formatted_inspirations = format_inspirations(inspirations)
            
            # 4. 获取硬件文档
            hardware_docs = get_hardware_doc(backend, arch)
            
            # 5. 渲染 User Prompt
            user_prompt = self.user_prompt_template.format(
                sketch_guide=self.sketch_guide,  # 关键！确保 LLM 输出 sketch DSL 格式
                skills=selected_skills,
                op_name=op_name,
                task_desc=remove_copyright_from_text(task_desc),
                inspirations=formatted_inspirations,
                conductor_suggestion=conductor_suggestion,
                user_requirements=user_requirements,
                hardware_docs=hardware_docs,
                arch_name=arch,
                enable_hint_mode=enable_hint_mode,
                has_hint=has_hint,
                format_instructions=self.format_instructions
            )
            
            # 6. 组合完整 prompt
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # 7. 创建 Jinja2 模板包装（用于 run_llm）
            template = Jinja2TemplateWrapper("{{ prompt }}")
            
            # 8. 更新上下文
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
            
            # 9. 调用 LLM
            llm_result, formatted_prompt, reasoning = await self.run_llm(
                template,
                {"prompt": full_prompt},
                model_level or "standard"
            )
            
            # 10. 处理 LLM 返回结果
            # ============ 处理 Hint 模式的输出 ============
            if enable_hint_mode and has_hint:
                try:
                    # 使用 robust 方法解析 JSON 格式的生成内容（支持 markdown 代码块包裹等多种格式）
                    extracted_json = ParserFactory._extract_json_comprehensive(llm_result)
                    if extracted_json:
                        result_dict = json.loads(extracted_json)
                    else:
                        # 尝试直接解析
                        result_dict = json.loads(llm_result)
                    
                    sketch = result_dict.get("sketch", "")
                    reasoning = result_dict.get("reasoning", reasoning)
                    
                    # 转换为标准格式（支持 parser_config.yaml 定义：code + 可选的 space_config_code）
                    # 将{"sketch": "...", "space_config": "...", "reasoning": "..."} 转换为 {"code": "...", "space_config_code": "..."}
                    result_for_return = {"code": sketch}
                    if "space_config" in result_dict:
                        space_config_code = result_dict["space_config"]
                        result_for_return["space_config_code"] = space_config_code
                        logger.info(f"[Task {task_id}] KernelDesigner 生成了参数空间配置")
                    
                    standard_result = json.dumps(result_for_return, ensure_ascii=False)
                    
                    # 返回: (标准格式的JSON字符串, 格式化提示词, 推理内容)
                    return standard_result, formatted_prompt, reasoning
                
                except json.JSONDecodeError as e:
                    # 如果解析失败，按原有流程返回
                    logger.warning(f"[{op_name}] Hint 模式下 JSON 解析失败: {e}，使用原始输出")
                    return llm_result, formatted_prompt, reasoning
            
            # ============ 非 Hint 模式：直接返回 LLM 结果（与原版 Designer 一致）============
            return llm_result, formatted_prompt, reasoning
        
        except Exception as e:
            logger.error(f"Exception in kernel_designer.run: {type(e).__name__}: {e}")
            raise
