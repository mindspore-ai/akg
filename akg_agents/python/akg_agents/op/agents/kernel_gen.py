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
from typing import Dict, Any, Tuple, List, Optional
from akg_agents import get_project_root
from akg_agents.core_v2.agents import AgentBase, register_agent, Jinja2TemplateWrapper
from akg_agents.core_v2.filesystem import ActionRecord

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
            }
        },
        "required": ["op_name", "task_desc", "dsl", "framework", "backend"]
    }
    
    def __init__(self):
        """初始化 KernelGen Agent"""
        
        self.codegen_step_count = 0
        
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
        """按 DSL 加载对应子目录的 skills，带缓存"""
        dsl_key = dsl.replace("_", "-")  # triton_ascend -> triton-ascend
        if dsl_key in self._skills_cache:
            return self._skills_cache[dsl_key]
        
        dsl_dir = SKILLS_DIR / dsl_key
        if not dsl_dir.exists():
            logger.warning(f"Skills directory not found for DSL '{dsl_key}': {dsl_dir}")
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
    
    async def _select_skills(
        self, 
        op_name: str = "",
        task_desc: str = "",
        dsl: str = "", 
        framework: str = "", 
        backend: str = "",
    ) -> List[Any]:
        """
        两阶段 Skill 选择：
        1. 按 DSL 加载对应子目录的 skills
        2. 粗筛：基于 OperatorSkillSelector（backend/dsl metadata 过滤）
        3. 精筛：LLM 根据任务描述选择
        """
        loaded_skills = self._load_skills_by_dsl(dsl)
        if not self.skill_selector or not loaded_skills:
            return []
        
        # [HACK] 算子类型精确匹配：attention 类只加载 attention skill，避免 prompt 膨胀
        ATTENTION_KEYWORDS = ["attention", "flash_attn", "sdpa", "scaled_dot_product", "gqa", "mqa"]
        combined_text = (op_name + " " + task_desc).lower()
        if any(kw in combined_text for kw in ATTENTION_KEYWORDS):
            attention_only = [s for s in loaded_skills if "attention" in s.name.lower()]
            if attention_only:
                logger.info(f"[KernelGen] Attention 算子检测到，仅加载 attention skill: {[s.name for s in attention_only]}")
                return attention_only
        
        try:
            # 阶段1：粗筛（使用 OperatorSkillSelector）
            context = OperatorSelectionContext(
                dsl=dsl.replace("_", "-"),  # triton_ascend -> triton-ascend
                backend=backend,
                include_category_groups=["knowledge"]  # guide, fundamental, dsl, method, implementation, reference
            )
            filtered = self.skill_selector.coarse_filter(loaded_skills, context)
            
            logger.info(f"Coarse filter: {len(loaded_skills)} -> {len(filtered)} skills")
            
            if len(filtered) <= 5:
                return filtered
            
            # 阶段2：LLM 精筛
            skills_info = [{"name": s.name, "description": s.description} for s in filtered]
            
            import json
            llm_prompt = f"""# Skill 智能筛选

你是一个专业的内核代码生成专家，需要根据当前任务特征，从候选的 Skills 中筛选出**所有相关**的知识文档。

## 当前任务

**算子名称**: {op_name}

**目标环境**:
- DSL: {dsl}
- 后端: {backend}
- 框架: {framework}

**任务描述**:
```
{task_desc}
```

## 候选 Skills

以下是所有可用的 Skills（共 {len(skills_info)} 个），包含名称和描述：

{json.dumps(skills_info, indent=2, ensure_ascii=False)}

## 筛选任务

请分析当前任务和所有候选 Skills，筛选出**所有对当前内核代码生成任务有参考价值**的 Skills。

### 筛选标准

**直接相关（必选）**：
- 算子模式匹配（如 elementwise、reduce、matmul、attention 等）
- DSL/后端相关的 API 参考或编程基础以及调试指南
- 优化策略直接适用于当前算子类型

**间接相关（可选）**：
- 包含部分相似的操作模式
- 优化技巧具有通用性，可迁移到当前任务
- 实现思路或代码结构有参考价值

**不相关（排除）**：
- 算子类型完全不同且无参考价值
- 与当前 DSL/后端不匹配
- 属于工作流、测试、草图设计等非代码生成类

### 筛选原则

1. **宁可多选，不要漏选**：如果某个 Skill 有任何参考价值，就应该被选中
2. **只排除明显不相关的**：只排除完全不相关的 Skills
3. **按相关性排序**：将最相关的 Skills 排在前面

## 输出要求

请输出 JSON 格式的筛选结果：

```json
{{
  "selected": ["skill-name-1", "skill-name-2", ...],
  "reason": "简要说明选择理由"
}}
```

**注意**：
- 返回完整的 Skill 名称，确保与上述候选 Skills 中的 name 完全一致
- 如果所有 Skills 都不相关，返回空列表
- selected 列表按相关性从高到低排序
"""
            template = Jinja2TemplateWrapper("{{ prompt }}")
            response, _, _ = await self.run_llm(template, {"prompt": llm_prompt}, "standard")
            
            # 解析响应
            selected_names, reason = self._parse_llm_selection(response)
            name_to_skill = {s.name: s for s in filtered}
            selected = [name_to_skill[n] for n in selected_names if n in name_to_skill]
            
            logger.info(f"LLM selected {len(selected)} skills: {[s.name for s in selected]}")
            if reason:
                logger.info(f"Selection reason: {reason}")
            return selected if selected else filtered
            
        except Exception as e:
            logger.warning(f"Skill selection failed: {e}")
            return []
    
    def _parse_llm_selection(self, response: str) -> Tuple[List[str], str]:
        """解析 LLM 返回的 skill 选择结果，返回 (名称列表, 理由)"""
        import json
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
    
    CATEGORY_ORDER = ["fundamental", "method", "implementation", "example"]
    
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
        previous_code: str = ""
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
        
        Returns:
            Tuple[str, str, str]: (生成的代码, 完整 prompt, 推理过程)
        """
        try:
            # 确保 history_compress 不为 None
            if history_compress is None:
                history_compress = []
            
            # 生成函数名
            func_name = f"{op_name}_{dsl}_{framework}"
            
            # 判断是否为纯修改模式（有前序代码 + 用户需求，且没有报错）
            is_pure_modification = bool(previous_code and user_requirements and not verifier_error)
            
            # 1. 选择相关 Skills
            if is_pure_modification:
                # 纯修改模式：用户需求明确，跳过 skill 选择，减少 prompt 长度
                selected_skills = []
                logger.info(f"[KernelGen] 纯修改模式：跳过 skill 选择，优先用户需求")
            else:
                # 首次生成 或 有报错需要修复：加载 skills 提供知识参考
                # 有 verifier_error 时尤其需要 skill 知识来修正代码
                selected_skills = await self._select_skills(
                    op_name=op_name,
                    task_desc=task_desc,
                    dsl=dsl, 
                    framework=framework, 
                    backend=backend
                )
            
            # 2. 按 category → name 排序并拼接 skill 内容
            skill_contents = self._assemble_skill_contents(selected_skills)
            
            # 3. 渲染 System Prompt
            system_prompt = self.system_prompt_template.format(
                dsl=dsl,
                framework=framework,
                backend=backend,
                arch=arch
            )
            
            # 4. 渲染 User Prompt
            user_prompt = self.user_prompt_template.format(
                history_actions=history_compress,
                verifier_error=verifier_error,
                conductor_suggestion=conductor_suggestion,
                skill_contents=skill_contents,
                op_name=op_name,
                func_name=func_name,
                task_desc=task_desc,
                user_requirements=user_requirements,
                previous_code=previous_code,
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
