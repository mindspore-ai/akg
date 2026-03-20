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
import re
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from akg_agents import get_project_root

from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper
from akg_agents.core_v2.filesystem import ActionRecord
from akg_agents.utils.common_utils import remove_copyright_from_text
from akg_agents.utils.hardware_utils import get_hardware_doc

from akg_agents.core_v2.skill import SkillLoader

# 设置 Skills 目录路径
project_root = Path(get_project_root())
SKILLS_DIR = project_root / "op" / "resources" / "skills"

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class KernelDesigner(AgentBase):
    """
    Kernel 算法草图设计 Agent
    
    基于 Skill 系统，根据任务需求动态选择知识和策略，生成高质量的算法草图。
    """
    
    # Agent 工具配置元数据
    TOOL_NAME = "call_kernel_design"
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
    
    def __init__(self):
        """初始化 KernelDesigner Agent"""
        
        self.design_step_count = 0
        
        context = {
            "agent_name": "kernel_designer",
            "task_label": "main",
        }
        
        super().__init__(context=context)
        
        # ==================== Skill 系统初始化 ====================
        self._init_skills()
        
        # ==================== Prompt 模板初始化 ====================
        self.system_prompt_template = self.load_template("kernel_designer/system_prompt.j2")
        self.user_prompt_template = self.load_template("kernel_designer/user_prompt.j2")
    
    def _init_skills(self):
        """初始化 Skill 系统（延迟加载，按 DSL 加载）"""
        self._loader = SkillLoader()
        self._skills_cache: Dict[str, list] = {}  # dsl -> skills
    
    def _load_skills_by_dsl(self, dsl: str) -> list:
        """加载 Designer 所需的固定 skills，带缓存

        加载范围仅 designer/ 目录：sketch-design、hint-mode 等。
        cases/（手写优化建议）由上层搜索控制器通过 SkillHandwriteLoader 加载，
        经采样后以 handwrite_suggestions 参数传入 run()。
        """
        dsl_key = dsl.replace("_", "-")
        if dsl_key in self._skills_cache:
            return self._skills_cache[dsl_key]

        skills = []
        designer_dir = SKILLS_DIR / "designer"
        if designer_dir.exists():
            try:
                designer_skills = self._loader.load_from_directory(designer_dir)
                skills.extend(designer_skills)
                logger.info(f"Loaded {len(designer_skills)} designer skills from {designer_dir}")
            except Exception as e:
                logger.error(f"Failed to load designer skills: {e}")

        self._skills_cache[dsl_key] = skills
        return skills
    
    async def _select_skills(self, dsl: str = "",
                             enable_hint_mode: bool = False,
                             has_hint: bool = False) -> List[Any]:
        """Designer 的 Skill 选择（仅固定 skills）

        1. 必选：sketch-design（对应旧 Designer 的 sketch_guide）
        2. 可选：hint-mode（当 enable_hint_mode 且 has_hint）
        """
        loaded_skills = self._load_skills_by_dsl(dsl)
        if not loaded_skills:
            return []

        try:
            skill_dict = {s.name: s for s in loaded_skills}
            selected_skills = []

            if "sketch-design" in skill_dict:
                selected_skills.append(skill_dict["sketch-design"])
                logger.info("Selected skill: sketch-design (required)")
            else:
                logger.warning("Skill 'sketch-design' not found in loaded skills, skipping required skill")

            if enable_hint_mode and has_hint and "hint-mode" in skill_dict:
                selected_skills.append(skill_dict["hint-mode"])
                logger.info("Selected skill: hint-mode (hint mode enabled)")
            else:
                logger.warning(
                    "Skill 'hint-mode' not applied: enable_hint_mode=%s, has_hint=%s, in_dict=%s",
                    enable_hint_mode, has_hint, "hint-mode" in skill_dict,
                )

            logger.info(f"Designer selected {len(selected_skills)} fixed skills: {[s.name for s in selected_skills]}")
            return selected_skills

        except Exception as e:
            logger.warning(f"Skill selection failed: {e}")
            return selected_skills if 'selected_skills' in locals() else []
    
    CATEGORY_ORDER = ["fundamental", "guide", "method", "implementation", "example"]
    
    def _assemble_skill_contents(self, selected_skills: List[Any]) -> str:
        """按 category -> name 排序 skills，拼接其内容为字符串。"""
        if not selected_skills:
            return ""
        
        def sort_key(skill):
            try:
                category_idx = self.CATEGORY_ORDER.index(skill.category or "")
            except ValueError:
                category_idx = 999
            return (category_idx, skill.name)
        
        sorted_skills = sorted(selected_skills, key=sort_key)
        
        order_desc = [
            f"{s.name}[{s.category or '?'}]"
            for s in sorted_skills
        ]
        logger.info(f"Skill assembly order: {order_desc}")
        
        return "\n\n---\n\n".join(skill.content for skill in sorted_skills)
    
    @staticmethod
    def _extract_sketch(raw_output: str) -> str:
        """从 LLM 输出中提取算法草图。
        
        预期 LLM 直接输出草图（无 markdown 包裹）。
        如果模型输出了代码块包裹，则做容错清理。
        """
        text = raw_output.strip()
        
        # 容错：提取 ```...``` 代码块中的内容（sketch 可能没有语言标签，或标为 text/plaintext）
        pattern = r'```(?:\w+)?\s*\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            text = max(matches, key=len).strip()
        
        return text
    
    @staticmethod
    def _parse_hint_output(raw_output: str) -> Tuple[str, Optional[str]]:
        """解析 Hint 模式的 JSON 输出，提取 code 和 space_config。
        
        Hint 模式仍使用 JSON 格式（需要同时输出 sketch 和 space_config）。
        
        Returns:
            Tuple[str, Optional[str]]: (sketch_code, space_config_code)
        """
        import json
        try:
            json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                code = result.get("code", "")
                space_config = result.get("space_config")
                return code, space_config
        except json.JSONDecodeError:
            pass
        return raw_output.strip(), None
    
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
        model_level: str = "standard",
        inspirations: str = "",
        handwrite_suggestions: Optional[List[Dict[str, str]]] = None,
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
            inspirations: 格式化后的进化探索方案字符串（evolve/adaptive_search 场景，
                          运行时动态生成的父代代码和性能数据，非静态知识，Skill 系统无法覆盖）
            handwrite_suggestions: 手写优化建议列表，由上层搜索控制器
                          通过 SkillHandwriteLoader + SkillHandwriteSampler 采样后传入。
                          每个 dict 包含 {name, improvement_doc}。
        
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
            
            # 1. 选择固定 Skills（sketch-design, hint-mode）
            selected_skills = await self._select_skills(
                dsl=dsl,
                enable_hint_mode=enable_hint_mode,
                has_hint=has_hint,
            )
            
            # 2. 按 category → name 排序并拼接 skill 内容
            skill_contents = self._assemble_skill_contents(selected_skills)
            
            # 3. 渲染 System Prompt
            system_prompt = self.system_prompt_template.format(
                dsl=dsl,
                backend=backend,
                arch=arch
            )
            
            # 4. 获取硬件文档
            hardware_docs = get_hardware_doc(backend, arch)
            
            # 5. 渲染 User Prompt
            user_prompt = self.user_prompt_template.format(
                history_actions=history_compress,
                skill_contents=skill_contents,
                op_name=op_name,
                task_desc=remove_copyright_from_text(task_desc),
                user_requirements=user_requirements,
                hardware_docs=hardware_docs,
                arch_name=arch,
                enable_hint_mode=enable_hint_mode,
                has_hint=has_hint,
                inspirations=inspirations,
                handwrite_suggestions=handwrite_suggestions or [],
            )
            
            # 6. 组合完整 prompt，非 Hint 模式追加输出格式硬约束
            if not has_hint:
                output_instruction = (
                    "\n\n## 输出格式（必须严格遵守）\n\n"
                    "请直接输出算法草图（sketch）。你的输出会被直接用于指导后续代码生成，因此：\n"
                    "- 不要使用 JSON 格式\n"
                    "- 不要使用 ```代码块``` 包裹\n"
                    "- 不要输出任何非草图内容（不要有解释文字、不要有 markdown 标记）\n"
                    "- 直接输出 `sketch op_name { ... }` 格式的算法草图\n"
                )
                full_prompt = f"{system_prompt}\n\n{user_prompt}{output_instruction}"
            else:
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
            raw_output, formatted_prompt, reasoning = await self.run_llm(
                template,
                {"prompt": full_prompt},
                model_level or "standard"
            )
            
            # 10. 提取草图
            if has_hint:
                sketch, space_config = self._parse_hint_output(raw_output)
                self._last_space_config = space_config
            else:
                sketch = self._extract_sketch(raw_output)
                self._last_space_config = None
            
            return sketch, formatted_prompt, reasoning
        
        except Exception as e:
            logger.error(f"Exception in kernel_designer.run: {type(e).__name__}: {e}")
            raise
