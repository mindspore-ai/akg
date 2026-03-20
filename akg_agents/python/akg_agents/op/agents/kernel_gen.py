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
import os
import re
import py_compile
import tempfile
from pathlib import Path

from akg_agents.core.utils import dsl_to_dir_key
from typing import Dict, Any, Tuple, List, Optional
from akg_agents import get_project_root
from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper
from akg_agents.core_v2.filesystem import ActionRecord
from akg_agents.op.utils.triton_ascend_api_docs import (
    resolve_triton_ascend_api_docs,
)
from akg_agents.utils.hardware_utils import get_hardware_doc

# 导入 skill 系统模块
from akg_agents.core_v2.skill import SkillLoader
# 使用算子专用的 SkillSelector
from akg_agents.op.skill.operator_selector import (
    OperatorSkillSelector,
    OperatorSelectionContext,
)

# 设置 Skills 目录路径
project_root = Path(get_project_root())
SKILLS_DIR = project_root / "op" / "resources" / "skills"

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class KernelGen(AgentBase):
    """
    Kernel 代码生成 Agent
    
    基于 Skill 系统，根据任务需求动态选择知识和策略，生成高性能内核代码。
    """
    
    # Agent 工具配置元数据
    TOOL_NAME = "call_kernel_gen"
    DESCRIPTION = """
仅生成 kernel 代码（不包含验证）。

功能：
- 根据任务描述生成高性能内核代码
- 基于 Skill 系统自动选择最佳优化策略
- 支持多种 DSL（Triton, CUDA, AscendC 等）

适用场景：
- 用户明确说"不用验证"、"只给我代码"、"快速生成草稿"
- 只需要快速查看代码实现思路
- 用户想自己进行验证和调试

⚠️ 注意：此工具仅生成代码，不验证正确性和性能！
- 如果需要验证代码正确性，请使用 verify_kernel
- 如果需要完整的生成+验证流程，请使用 workflow（如 use_coder_only_workflow）

输出：生成的 kernel 代码（包含 class ModelNew 和 kernel 函数）
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
            "verifier_error": { 
                "type": "string",
                "description": "Verifier 错误信息（可选）",
                "default": ""
            },
            "conductor_suggestion": {
                "type": "string",
                "description": "Conductor 修复建议（可选）",
                "default": ""
            },
            "model_level": {
                "type": "string",
                "description": "模型级别（例如：'standard', 'fast', 'complex'）",
                "default": "standard"
            },
            "previous_code": {
                "type": "string",
                "description": "之前生成的 kernel 代码（用于修改优化场景），包含 class ModelNew 等。如果提供，会基于此代码进行修改而非从零生成",
                "default": ""
            },
            "designer_code": {
                "type": "string",
                "description": "KernelDesigner 生成的算法草图/伪代码（可选，DefaultWorkflowV2 场景使用）。提供后会以此为基础生成代码",
                "default": ""
            },
            "inspirations": {
                "type": "string",
                "description": "格式化后的进化探索方案字符串（可选，evolve/adaptive_search 场景使用）",
                "default": ""
            },
            "handwrite_suggestions": {
                "type": "array",
                "description": "手写优化策略与实现参考（可选）",
                "items": {
                    "type": "object"
                },
                "default": []
            },
            "extra_skills": {
                "type": "array",
                "description": "额外注入的 skill 对象列表（可选，跳过筛选，精筛后直接追加）",
                "items": {
                    "type": "object"
                },
                "default": []
            }
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend"]
    }
    
    def __init__(self):
        """初始化 KernelGen Agent"""
        
        self.codegen_step_count = 0
        self.format_instructions = ""
        self.extra_skills: List[Any] = []
        
        context = {
            "agent_name": "kernel_gen",
            "task_label": "main",
        }
        
        super().__init__(context=context)
        
        # ==================== Skill 系统初始化 ====================
        self._init_skills()
        
        # ==================== Prompt 模板初始化 ====================
        self.system_prompt_template = self.load_template("kernel_gen/system_prompt.j2")
        self.user_prompt_template = self.load_template("kernel_gen/user_prompt.j2")
    
    def _init_skills(self):
        """初始化 Skill 系统（延迟加载，按 DSL 加载）"""
        self._loader = SkillLoader()
        self._skills_cache: Dict[str, list] = {}  # dsl -> skills
        # 使用算子专用的 OperatorSkillSelector
        self.skill_selector = OperatorSkillSelector()
    
    def _load_skills_by_dsl(self, dsl: str) -> list:
        """加载 {dsl}/ 整个目录下的 skills（guides + cases + evolved），带缓存"""
        dsl_key = dsl_to_dir_key(dsl)
        if dsl_key in self._skills_cache:
            return self._skills_cache[dsl_key]
        
        dsl_dir = SKILLS_DIR / dsl_key
        if not dsl_dir.exists():
            logger.warning(f"Skill directory not found for DSL '{dsl_key}': {dsl_dir}")
            self._skills_cache[dsl_key] = []
            return []
        
        try:
            skills = self._loader.load_from_directory(dsl_dir)
            self._skills_cache[dsl_key] = skills
            logger.info(f"Loaded {len(skills)} skills from {dsl_dir}")
            return skills
        except Exception as e:
            logger.error(f"Failed to load skills for DSL '{dsl_key}': {e}")
            self._skills_cache[dsl_key] = []
            return []
    
    # ==================== 分层分阶段 Skill 选择 ====================

    STAGE_CATEGORIES = {
        "initial":  {"fundamental", "reference", "guide", "example"},
        "debug":    {"fundamental", "reference", "guide", "example", "case"},
        "optimize": {"fundamental", "reference", "guide", "example", "case"},
    }

    async def _select_skills_by_stage(
        self,
        op_name: str,
        task_desc: str,
        dsl: str,
        framework: str,
        backend: str,
        stage: str,
        verifier_error: str = "",
    ) -> List[Any]:
        """分层分阶段 Skill 选择：
        Layer 0 (fundamental/reference): 固定注入
        Layer 1 (guide + example): LLM 一次筛选 guide(1-2 个)，example 跟随 guide 的 operator_type
        Layer 2 (case): 仅 debug/optimize 阶段，与 guide 一起在同一次 LLM 调用中筛选
        """
        all_skills = self._load_skills_by_dsl(dsl)
        if not all_skills:
            return []

        allowed = self.STAGE_CATEGORIES.get(stage, self.STAGE_CATEGORIES["initial"])

        always_skills: List[Any] = []
        guide_candidates: List[Any] = []
        example_candidates: List[Any] = []
        case_fix: List[Any] = []
        case_improve: List[Any] = []

        for skill in all_skills:
            cat = getattr(skill, "category", "") or ""
            if cat not in allowed:
                continue
            if cat in ("fundamental", "reference"):
                always_skills.append(skill)
            elif cat == "guide":
                guide_candidates.append(skill)
            elif cat == "example":
                example_candidates.append(skill)
            elif cat == "case":
                ct = self._infer_case_type(skill)
                if ct == "fix":
                    case_fix.append(skill)
                else:
                    case_improve.append(skill)

        if stage == "debug":
            case_candidates = case_fix
        elif stage == "optimize":
            case_candidates = case_improve
        else:
            case_candidates = []

        guide_selected, case_selected = await self._llm_select_guides_and_cases(
            guide_candidates=guide_candidates,
            case_candidates=case_candidates,
            op_name=op_name,
            task_desc=task_desc,
            verifier_error=verifier_error,
            stage=stage,
        )

        selected_types = set()
        for s in guide_selected:
            meta = getattr(s, "metadata", {}) or {}
            ot = meta.get("operator_type", "")
            if ot:
                selected_types.add(ot)

        example_selected = []
        for s in example_candidates:
            meta = getattr(s, "metadata", {}) or {}
            ex_type = meta.get("operator_type", "")
            fw = meta.get("framework", "all")
            if ex_type and ex_type in selected_types:
                if not framework or framework in fw or fw == "all":
                    example_selected.append(s)
            elif not ex_type and (not framework or framework in fw or fw == "all"):
                example_selected.append(s)

        result = always_skills + guide_selected + example_selected + case_selected
        logger.info(
            f"[KernelGen] stage={stage}, "
            f"L0={len(always_skills)}, L1_guide={len(guide_selected)}, "
            f"L1_example={len(example_selected)}, L2_case={len(case_selected)} "
            f"(fix_pool={len(case_fix)}, improve_pool={len(case_improve)}), "
            f"total_chars={sum(len(s.content) for s in result)}"
        )
        excluded_guides = [s.name for s in guide_candidates if s not in guide_selected]
        excluded_cases = [s.name for s in case_candidates if s not in case_selected]
        if excluded_guides:
            logger.info(f"[KernelGen] 排除的 guide: {excluded_guides}")
        if excluded_cases:
            logger.info(f"[KernelGen] 排除的 case: {excluded_cases}")
        return result

    @staticmethod
    def _infer_case_type(skill) -> str:
        """推断 case skill 类型：fix（错误修复）或 improvement（性能优化）。
        优先级：metadata.case_type > metadata.source > 目录名推断 > 默认 improvement。
        """
        meta = getattr(skill, "metadata", {}) or {}
        ct = meta.get("case_type", "")
        if ct in ("fix", "improvement"):
            return ct
        source = meta.get("source", "")
        if source == "error_fix":
            return "fix"
        skill_path = getattr(skill, "skill_path", None)
        if skill_path:
            path_str = str(skill_path)
            if "evolved-fix" in path_str:
                return "fix"
            if "evolved-improvement" in path_str:
                return "improvement"
        return "improvement"

    async def _llm_select_guides_and_cases(
        self,
        guide_candidates: List[Any],
        case_candidates: List[Any],
        op_name: str,
        task_desc: str,
        verifier_error: str,
        stage: str,
    ) -> tuple:
        """一次 LLM 调用同时筛选 guide 和 case skills。
        返回 (guide_selected, case_selected)。
        """
        import json

        if not guide_candidates and not case_candidates:
            logger.info("[KernelGen] guide 和 case 候选均为空，跳过 LLM 筛选")
            return [], []

        logger.debug(
            f"[KernelGen] 开始 LLM 筛选: "
            f"guide 候选={[s.name for s in guide_candidates]}, "
            f"case 候选={[s.name for s in case_candidates]}"
        )

        guide_info = [{"name": s.name, "description": s.description} for s in guide_candidates]
        case_info = [{"name": s.name, "description": s.description} for s in case_candidates]

        task_desc_truncated = task_desc[:1500] if len(task_desc) > 1500 else task_desc

        sections = [f"**算子名称**: {op_name}\n**任务描述**:\n```python\n{task_desc_truncated}\n```"]

        if verifier_error:
            sections.append(f"**当前错误**:\n```\n{verifier_error[:500]}\n```")

        if guide_info:
            sections.append(
                f"## 算子优化指南（guide）\n"
                f"从以下指南中选出与当前算子**最相关的 1 个**（若算子融合了两种不同的计算模式，最多选 2 个）。"
                f"**只选最大重叠的，不要多选。**\n\n"
                f"{json.dumps(guide_info, indent=2, ensure_ascii=False)}"
            )

        if case_info:
            if stage == "debug":
                case_instruction = (
                    "从以下案例中选出与**当前错误强相关**的案例。"
                    "优先选处理过相同类型错误的，其次选相同算子类型的。不相关的不选。"
                )
            else:
                case_instruction = (
                    "从以下案例中选出与**当前算子优化目标强相关**的案例。"
                    "优先选相同算子类型的优化案例，其次选相似优化策略的。不相关的不选。"
                )
            sections.append(
                f"## 优化/修复案例（case）\n{case_instruction}\n\n"
                f"{json.dumps(case_info, indent=2, ensure_ascii=False)}"
            )

        response_format = '{"guides": ["guide-name-1"], "cases": ["case-name-1", ...], "reason": "选择理由"}'
        if not case_info:
            response_format = '{"guides": ["guide-name-1"], "reason": "选择理由"}'

        llm_prompt = (
            f"# Skill 筛选\n\n"
            + "\n\n".join(sections)
            + f"\n\n## 返回格式\n仅返回 JSON，不要有其他文字:\n{response_format}"
        )

        try:
            template = Jinja2TemplateWrapper("{{ prompt }}")
            response, _, _ = await self.run_llm(template, {"prompt": llm_prompt}, "fast")
            logger.debug(f"[KernelGen] LLM 筛选原始响应: {response[:500]}")

            parsed = self._parse_unified_selection(response)
            if not parsed:
                logger.warning(f"[KernelGen] LLM 返回无法解析为 JSON: {response[:300]}")
                return [], []

            guide_names = parsed.get("guides", [])
            case_names = parsed.get("cases", [])
            reason = parsed.get("reason", "")

            guide_map = {s.name: s for s in guide_candidates}
            case_map = {s.name: s for s in case_candidates}
            guide_selected = [guide_map[n] for n in guide_names if n in guide_map]
            case_selected = [case_map[n] for n in case_names if n in case_map]

            unmatched_guides = [n for n in guide_names if n not in guide_map]
            if unmatched_guides:
                logger.warning(f"[KernelGen] LLM 返回的 guide 名称无法匹配: {unmatched_guides}")

            logger.info(
                f"[KernelGen] LLM 筛选结果 (stage={stage}): "
                f"guide={[s.name for s in guide_selected]}, "
                f"case={[s.name for s in case_selected]}"
            )
            if reason:
                logger.info(f"[KernelGen] 筛选理由: {reason}")

            if not guide_selected and guide_candidates:
                logger.info("[KernelGen] LLM 未选中任何 guide，不注入 guide 层知识")

            return guide_selected, case_selected
        except Exception as e:
            logger.warning(f"[KernelGen] LLM skill selection failed: {e}")
            return [], []

    def _parse_unified_selection(self, response: str) -> dict:
        """解析 LLM 返回的统一筛选结果 JSON。
        支持 markdown 代码块包裹、嵌套数组等。
        """
        import json
        import re

        cleaned = response.strip()
        fence = re.search(r'```(?:json)?\s*\n?(.*?)```', cleaned, re.DOTALL)
        if fence:
            cleaned = fence.group(1).strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        brace_start = cleaned.find('{')
        if brace_start >= 0:
            depth = 0
            for i in range(brace_start, len(cleaned)):
                if cleaned[i] == '{':
                    depth += 1
                elif cleaned[i] == '}':
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(cleaned[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break
        return {}
    
    CATEGORY_LAYER = {
        "fundamental": (0, 0), "reference": (0, 1),
        "guide": (1, 0), "example": (1, 1),
        "case": (2, 0),
    }

    def _sort_key(self, skill):
        cat = getattr(skill, "category", "") or ""
        layer, sublayer = self.CATEGORY_LAYER.get(cat, (9, 9))
        return (layer, sublayer, getattr(skill, "name", ""))

    def _assemble_skill_contents(self, selected_skills: List[Any]) -> str:
        """按 CATEGORY_LAYER 排序，分区组装 skill 内容。"""
        if not selected_skills:
            return ""

        sorted_skills = sorted(selected_skills, key=self._sort_key)

        fundamentals = [s for s in sorted_skills
                        if (getattr(s, "category", "") or "") in ("fundamental", "reference")]
        guides = [s for s in sorted_skills
                  if (getattr(s, "category", "") or "") == "guide"]
        examples = [s for s in sorted_skills
                    if (getattr(s, "category", "") or "") == "example"]
        cases = [s for s in sorted_skills
                 if (getattr(s, "category", "") or "") == "case"]

        sections = []
        if fundamentals:
            content = "\n\n---\n\n".join(s.content for s in fundamentals)
            sections.append(f"### 基础知识与规范\n\n{content}")
        if guides:
            content = "\n\n---\n\n".join(s.content for s in guides)
            sections.append(f"### 算子优化指南\n\n{content}")
        if examples:
            content = "\n\n---\n\n".join(s.content for s in examples)
            sections.append(f"### 代码示例参考\n\n{content}")
        if cases:
            content = "\n\n---\n\n".join(s.content for s in cases)
            sections.append(f"### 优化/修复案例\n\n{content}")

        order_desc = [f"{s.name}[{getattr(s, 'category', '?')}]" for s in sorted_skills]
        logger.info(f"Skill assembly order: {order_desc}")

        return "\n\n---\n\n".join(sections)

    async def _load_aggregated_api_docs(self, dsl: str, backend: str = "", arch: str = "") -> str:
        """按需加载 Triton Ascend API 文档。"""
        if dsl != "triton_ascend":
            return ""

        return await resolve_triton_ascend_api_docs(backend=backend, arch=arch)
    
    @staticmethod
    def _extract_code(raw_output: str) -> str:
        """从 LLM 输出中提取纯 Python 代码。
        
        预期 LLM 直接输出纯代码（无 markdown 包裹）。
        如果模型仍然输出了 ```python``` 包裹，则做容错清理。
        """
        code = raw_output.strip()
        
        # 容错：如果模型还是用了 ```python ... ``` 包裹，提取内部内容
        pattern = r'```(?:python)?\s*\n(.*?)```'
        matches = re.findall(pattern, code, re.DOTALL)
        if matches:
            code = max(matches, key=len).strip()
        
        return code
    
    @staticmethod
    def _validate_syntax(code: str) -> Tuple[bool, str]:
        """使用 py_compile 验证 Python 代码语法。
        
        Returns:
            Tuple[bool, str]: (是否通过, 错误信息)
        """
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix='.py', mode='w', delete=False, encoding='utf-8'
            ) as f:
                f.write(code)
                tmp_path = f.name
            py_compile.compile(tmp_path, doraise=True)
            return True, ""
        except py_compile.PyCompileError as e:
            return False, str(e)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
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
        verifier_error: str = "",
        conductor_suggestion: str = "",
        model_level: str = "standard",
        previous_code: str = "",
        designer_code: str = "",
        inspirations: str = "",
        handwrite_suggestions: Optional[List[Dict[str, str]]] = None,
        extra_skills: Optional[List[Any]] = None,
    ) -> Tuple[str, str, str]:
        """
        执行代码生成
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述
            dsl: 目标 DSL
            framework: 目标框架
            backend: 目标后端
            arch: 目标架构
            user_requirements: 用户额外需求
            task_id: 任务 ID
            history_compress: 历史记录（已废弃，保留向后兼容）
            verifier_error: Verifier 错误信息
            conductor_suggestion: Conductor 修复建议
            model_level: 模型级别
            previous_code: 之前生成的代码（修改场景使用）
            designer_code: KernelDesigner 生成的算法草图（可选，DefaultWorkflowV2 场景使用）
            inspirations: 格式化后的进化探索方案字符串（evolve/kernelgen_only 场景使用）
            handwrite_suggestions: 手写优化策略与实现参考
            extra_skills: 额外注入的 skill 列表，跳过粗筛和精筛，在精筛结果后直接追加
        
        Returns:
            Tuple[str, str, str]: (生成的代码, 完整 prompt, 推理过程)
        """
        try:
            if history_compress is None:
                history_compress = []
            
            func_name = f"{op_name}_{dsl}_{framework}"
            is_pure_modification = bool(previous_code and user_requirements and not verifier_error)
            
            # Skill 选择
            if is_pure_modification:
                selected_skills = []
                logger.info("[KernelGen] 跳过 skill 加载: 纯修改模式（用户需求明确 + 无报错）")
            else:
                if verifier_error:
                    stage = "debug"
                elif inspirations:
                    stage = "optimize"
                else:
                    stage = "initial"
                selected_skills = await self._select_skills_by_stage(
                    op_name=op_name, task_desc=task_desc,
                    dsl=dsl, framework=framework, backend=backend,
                    stage=stage, verifier_error=verifier_error,
                )
            
            # 1.5 追加额外注入的 skills（跳过筛选，直接并入）
            all_extra = (extra_skills or []) + self.extra_skills
            if all_extra:
                existing_names = {s.name for s in selected_skills}
                appended = [s for s in all_extra if s.name not in existing_names]
                if appended:
                    selected_skills = selected_skills + appended
                    logger.info(f"[KernelGen] 追加 {len(appended)} 个 extra_skills: {[s.name for s in appended]}")

            # 2. 按 category → name 排序并拼接 skill 内容
            skill_contents = self._assemble_skill_contents(selected_skills)
            
            # 3. 渲染 System Prompt
            system_prompt = self.system_prompt_template.format(
                dsl=dsl,
                framework=framework,
                backend=backend,
                arch=arch
            )

            aggregated_api_docs = await self._load_aggregated_api_docs(
                dsl,
                backend=backend,
                arch=arch,
            )
            
            # 4. 渲染 User Prompt（verifier_error 只保留尾部关键信息）
            error_for_prompt = verifier_error
            if verifier_error and len(verifier_error) > 4000:
                error_for_prompt = "... (前面省略) ...\n" + verifier_error[-4000:]

            aggregated_api_docs = await self._load_aggregated_api_docs(
                dsl,
                backend=backend,
                arch=arch,
            )
            user_prompt = self.user_prompt_template.format(
                history_actions=history_compress,
                verifier_error=error_for_prompt,
                conductor_suggestion=conductor_suggestion,
                skill_contents=skill_contents,
                aggregated_api_docs=aggregated_api_docs,
                op_name=op_name,
                func_name=func_name,
                task_desc=task_desc,
                user_requirements=user_requirements,
                hardware_docs=get_hardware_doc(backend, arch),
                previous_code=previous_code,
                format_instructions=self.format_instructions,
                designer_code=designer_code,
                inspirations=inspirations,
                handwrite_suggestions=handwrite_suggestions or [],
                framework=framework,
                dsl=dsl,
            )
            
            # 5. 组合完整 prompt，末尾追加输出格式硬约束
            output_instruction = (
                "\n\n## 输出格式（必须严格遵守）\n\n"
                "请直接输出纯 Python 代码。你的输出会被直接保存为 .py 文件并执行，因此：\n"
                "- 不要使用 ```python``` 代码块包裹\n"
                "- 不要使用 JSON 格式\n"
                "- 不要输出任何非代码内容（不要有解释文字、不要有 markdown 标记）\n"
                "- 相关的思考分析，请以 Python 注释的方式写在代码中\n"
                "- 确保代码完整可执行，包含所有 import 语句和 class 定义\n"
            )
            full_prompt = f"{system_prompt}\n\n{user_prompt}{output_instruction}"
            
            # 6. 创建 Jinja2 模板包装（用于 run_llm）
            template = Jinja2TemplateWrapper("{{ prompt }}")
            
            # 7. 更新上下文
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
            
            # 8. 调用 LLM
            raw_output, formatted_prompt, reasoning = await self.run_llm(
                template,
                {"prompt": full_prompt},
                model_level or "standard"
            )
            
            # 9. 从 LLM 输出中提取纯代码
            generated_code = self._extract_code(raw_output)
            
            # 10. 使用 py_compile 验证语法
            syntax_ok, syntax_error = self._validate_syntax(generated_code)
            if syntax_ok:
                logger.info(f"[KernelGen] py_compile 语法校验通过")
            else:
                logger.warning(f"[KernelGen] py_compile 语法校验失败: {syntax_error}")
            
            return generated_code, formatted_prompt, reasoning
        
        except Exception as e:
            logger.error(f"Exception in kernel_gen.run: {type(e).__name__}: {e}")
            raise
