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
通用 LLM 驱动的 Skill 选择器

两阶段筛选：
1. 粗筛：基于 metadata 标签快速过滤
2. 精筛：使用 LLM 根据任务和历史智能选择

架构设计：
- SelectionContext: 通用选择上下文（领域无关）
- SkillSelector: 通用选择器，支持自定义过滤器（可插拔）

扩展方式：
- 继承 SelectionContext 添加领域特定字段
- 实现自定义过滤器函数
- 继承 SkillSelector 实现领域特定选择器

示例：
    # 文档生成场景
    context = SelectionContext(
        task_type="document_generation",
        custom_fields={"doc_type": "api", "language": "python"}
    )
    
    # 自定义过滤器
    def doc_filter(skill, context):
        doc_type = context.custom_fields.get("doc_type")
        return skill.metadata.get("doc_types", "").split(",") if doc_type else True
    
    selector = SkillSelector(custom_filters=[doc_filter])
    selected = selector.select(skills, context, llm_func)
"""

from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
import logging

from .metadata import SkillMetadata, CATEGORY_GROUPS

logger = logging.getLogger(__name__)


def _resolve_include_categories(context: "SelectionContext") -> set:
    """解析 context 中的 include_categories 与 include_category_groups，返回允许的 category 集合"""
    raw = getattr(context, "include_categories", None)
    if raw is None:
        raw = context.custom_fields.get("include_categories")
    include = set(raw or [])

    groups = getattr(context, "include_category_groups", None)
    if groups is None:
        groups = context.custom_fields.get("include_category_groups")
    for g in (groups or []):
        include.update(CATEGORY_GROUPS.get(g, []))
    return include


def _resolve_exclude_categories(context: "SelectionContext") -> set:
    """解析 context 中的 exclude_categories 与 exclude_category_groups，返回排除的 category 集合"""
    raw = getattr(context, "exclude_categories", None)
    if raw is None:
        raw = context.custom_fields.get("exclude_categories")
    exclude = set(raw or [])

    groups = getattr(context, "exclude_category_groups", None)
    if groups is None:
        groups = context.custom_fields.get("exclude_category_groups")
    for g in (groups or []):
        exclude.update(CATEGORY_GROUPS.get(g, []))
    return exclude


# ==================== 通用过滤器工具 ====================


def create_metadata_matcher(
    context_field: str,
    metadata_field: Optional[str] = None,
    match_mode: str = "include"
) -> Callable:
    """
    创建通用的 metadata 匹配器（工厂函数）
    
    这是一个高阶函数，用于生成过滤器函数。避免为每个字段重复编写相同的过滤逻辑。
    
    Args:
        context_field: context 中的字段名（如 "backend", "dsl", "hardware"）
        metadata_field: Skill metadata 中的字段名（如果为 None，则与 context_field 相同）
        match_mode: 匹配模式
            - "include": 必须包含（默认）- Skill 的 metadata 值必须包含 context 的值
            - "exclude": 必须排除 - Skill 的 metadata 值不能包含 context 的值
    
    Returns:
        过滤器函数 filter(skill, context) -> bool
    
    示例：
        # 创建 backend 匹配器
        backend_filter = create_metadata_matcher("backend")
        
        # 创建 operator_type 匹配器（metadata 字段名不同）
        operator_filter = create_metadata_matcher("operator_type", "operator_patterns")
        
        # 创建排除型匹配器
        exclude_cpu = create_metadata_matcher("backend", "backend", "exclude")
        
        # 组合使用：必须是 cuda 后端，但不能是 triton DSL
        filters = [
            create_metadata_matcher("backend"),  # 必须匹配 backend
            create_metadata_matcher("dsl", "dsl", "exclude")  # 排除某些 dsl
        ]
        selector = SkillSelector(custom_filters=filters)
    """
    if metadata_field is None:
        metadata_field = context_field
    
    def matcher(skill: SkillMetadata, context: SelectionContext) -> bool:
        # 尝试从 context 获取字段值（支持直接字段或 custom_fields）
        context_value = getattr(context, context_field, None)
        if context_value is None:
            context_value = context.custom_fields.get(context_field)
        
        if not context_value:
            return True  # 如果 context 没有指定该字段，放行
        
        # 获取 Skill metadata 中的值
        if not skill.metadata:
            return True  # 如果 Skill 没有 metadata，默认放行（可能是通用 Skill）
        
        metadata_value = skill.metadata.get(metadata_field, "")
        if not metadata_value:
            return True  # 如果 Skill 没有该 metadata 字段，默认放行
        
        # 解析逗号分隔的值列表
        values = [v.strip() for v in metadata_value.split(",")]
        
        # 根据匹配模式进行判断
        if match_mode == "include":
            # 特殊处理：如果 metadata 值包含 "all"，表示适用于所有情况
            if "all" in values:
                return True
            return context_value in values
        elif match_mode == "exclude":
            return context_value not in values
        else:
            raise ValueError(f"Unknown match_mode: {match_mode}")
    
    return matcher


def and_filters(*filters: Callable) -> Callable:
    """
    组合多个过滤器（AND 逻辑）
    
    所有过滤器都必须返回 True，最终结果才为 True。
    
    Args:
        *filters: 多个过滤器函数
    
    Returns:
        组合后的过滤器函数
    
    示例：
        # backend 必须是 cuda，且 dsl 必须是 triton
        combined = and_filters(
            create_metadata_matcher("backend"),
            create_metadata_matcher("dsl")
        )
        selector = SkillSelector(custom_filters=[combined])
    """
    def combined_filter(skill: SkillMetadata, context: SelectionContext) -> bool:
        return all(f(skill, context) for f in filters)
    return combined_filter


def or_filters(*filters: Callable) -> Callable:
    """
    组合多个过滤器（OR 逻辑）
    
    任一过滤器返回 True，最终结果就为 True。
    
    Args:
        *filters: 多个过滤器函数
    
    Returns:
        组合后的过滤器函数
    
    示例：
        # backend 是 cuda 或 ascend 都可以
        combined = or_filters(
            lambda s, c: s.metadata.get("backend") == "cuda",
            lambda s, c: s.metadata.get("backend") == "ascend"
        )
        selector = SkillSelector(custom_filters=[combined])
    """
    def combined_filter(skill: SkillMetadata, context: SelectionContext) -> bool:
        return any(f(skill, context) for f in filters)
    return combined_filter


def create_category_filter(match_mode: str = "include") -> Callable:
    """
    创建 Category 过滤器（工厂函数）
    
    支持两种模式：
    - include: 只包含指定的 categories 或 category_groups（白名单）
    - exclude: 排除指定的 categories 或 category_groups（黑名单）
    
    Args:
        match_mode: 匹配模式
            - "include": 只包含 context.include_categories / include_category_groups 中的分类
            - "exclude": 排除 context.exclude_categories / exclude_category_groups 中的分类
    
    Returns:
        过滤器函数 filter(skill, context) -> bool
    
    示例：
        # 创建 include 过滤器（只要 guide, example 的 Skill）
        include_filter = create_category_filter("include")
        context = SelectionContext(include_categories=["guide", "example"])
        
        # 使用分组：只要 knowledge 组
        context = SelectionContext(include_category_groups=["knowledge"])
        
        # 创建 exclude 过滤器（不要 workflow, agent）
        exclude_filter = create_category_filter("exclude")
        context = SelectionContext(exclude_categories=["workflow", "agent"])
    """
    def category_filter(skill: SkillMetadata, context: "SelectionContext") -> bool:
        if match_mode == "include":
            allowed = _resolve_include_categories(context)
            if not allowed:
                return True
            if not skill.category:
                return False  # 没有 category 的 Skill 不通过 include 筛选
            return skill.category in allowed
        
        elif match_mode == "exclude":
            excluded = _resolve_exclude_categories(context)
            if not excluded:
                return True
            if not skill.category:
                return True  # 没有 category 的 Skill 默认通过 exclude 筛选
            return skill.category not in excluded
        
        else:
            raise ValueError(f"Unknown match_mode: {match_mode}")
    
    return category_filter


# ==================== 选择上下文 ====================


@dataclass
class SelectionContext:
    """通用 Skill 选择上下文（领域无关）
    
    这是一个完全通用的基类，不包含任何领域特定的字段。
    所有领域特定的字段应该通过继承添加（如 OperatorSelectionContext）。
    
    字段说明：
        include_categories: Category 白名单，只包含这些 category 的 Skill
        exclude_categories: Category 黑名单，排除这些 category 的 Skill
        include_category_groups: 按分组白名单（如 ["knowledge"] 对应 guide/fundamental/dsl 等）
        exclude_category_groups: 按分组黑名单
        custom_fields: 自定义字段字典，用于传递任意信息
    
    示例：
        # 方式 1: 使用 custom_fields（适合简单场景）
        context = SelectionContext(
            custom_fields={
                "doc_type": "api",
                "language": "python"
            }
        )
        
        # 方式 2: 使用 Category 筛选
        context = SelectionContext(
            include_categories=["guide", "example"],
            exclude_categories=["workflow"]
        )
        
        # 方式 3: 使用分组
        context = SelectionContext(include_category_groups=["knowledge"])
        
        # 方式 4: 继承创建领域特定上下文
        @dataclass
        class DocSelectionContext(SelectionContext):
            doc_type: Optional[str] = None
            language: Optional[str] = None
        
        context = DocSelectionContext(doc_type="api", language="python")
    """
    # Category 筛选字段
    include_categories: Optional[List[str]] = None       # 白名单：只包含这些 category
    exclude_categories: Optional[List[str]] = None      # 黑名单：排除这些 category
    include_category_groups: Optional[List[str]] = None # 白名单分组（orchestration, actor, knowledge, example）
    exclude_category_groups: Optional[List[str]] = None # 黑名单分组
    
    # 自定义扩展字段（用于任意信息）
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（包含所有非私有字段）"""
        result = {}
        
        # 收集所有 dataclass 字段（除了 custom_fields）
        for field_name in dir(self):
            if not field_name.startswith('_') and field_name != 'custom_fields':
                value = getattr(self, field_name, None)
                # 只包含非方法的字段
                if not callable(value) and value is not None:
                    result[field_name] = value
        
        # 添加 custom_fields 中的内容
        result.update(self.custom_fields)
        
        return {k: v for k, v in result.items() if v is not None}


class SkillSelector:
    """
    通用 Skill 选择器
    
    提供两阶段筛选：
    1. 粗筛（coarse_filter）：基于 metadata 快速过滤
    2. 精筛（fine_select）：使用 LLM 智能选择
    
    支持自定义过滤器，可适配不同领域：
    - 算子生成：使用 OperatorFilterMixin
    - 文档生成：自定义 doc_type 过滤器
    - 测试生成：自定义 test_type 过滤器
    """
    
    def __init__(self, custom_filters: Optional[List[Callable]] = None):
        """初始化选择器
        
        Args:
            custom_filters: 自定义过滤器列表，每个过滤器签名为：
                filter_func(skill: SkillMetadata, context: SelectionContext) -> bool
        """
        self.custom_filters = custom_filters or []
    
    def coarse_filter(
        self,
        skills: List[SkillMetadata],
        context: SelectionContext,
        category: Optional[str] = None
    ) -> List[SkillMetadata]:
        """
        阶段 1：粗筛（基于 metadata 标签）
        
        快速过滤掉明显不相关的 Skill，减少 LLM 输入 Token
        
        Args:
            skills: 所有候选 Skill
            context: 选择上下文
            category: 指定只从某个分类筛选（可选）
        
        Returns:
            过滤后的候选 Skill 列表
        """
        candidates = []
        
        for skill in skills:
            # 通用过滤：分类匹配
            if category and skill.category != category:
                continue
            
            # 应用自定义过滤器
            if self.custom_filters:
                passed = True
                for filter_func in self.custom_filters:
                    if not filter_func(skill, context):
                        passed = False
                        break
                if not passed:
                    continue
            
            candidates.append(skill)
        
        logger.info(f"粗筛: {len(skills)} -> {len(candidates)} skills")
        return candidates
    
    def build_llm_prompt(
        self,
        candidates: List[SkillMetadata],
        context: SelectionContext,
        prompt_template: str,
        **kwargs
    ) -> str:
        """
        构建 LLM 精筛的 prompt
        
        用户必须提供 prompt 模板，框架负责准备通用的模板变量。
        
        Args:
            candidates: 粗筛后的候选 Skill
            context: 选择上下文
            prompt_template: prompt 模板（必需）
                模板中可以使用以下预定义变量：
                - {context_str}: 格式化的上下文信息
                - {skills_str}: 格式化的候选 Skill 列表
                - {candidates}: 候选 Skill 对象列表
                - {context}: 上下文对象
            **kwargs: 额外参数（用户自定义的任意参数），会传递给模板
                例如：history、max_skills、system_prompt 等
        
        Returns:
            LLM prompt 字符串
        
        Raises:
            ValueError: 如果未提供 prompt_template
            KeyError: 如果模板中使用了未提供的变量
        
        示例：
            # 简单模板
            template = '''
            上下文: {context_str}
            候选 Skills: {skills_str}
            请选择最相关的 Skills，返回 JSON 格式。
            '''
            prompt = selector.build_llm_prompt(candidates, context, template)
            
            # 使用额外参数
            template = '''
            上下文: {context_str}
            候选: {skills_str}
            历史: {history}
            请选择最多 {max_skills} 个 Skill。
            '''
            prompt = selector.build_llm_prompt(
                candidates, context, template,
                history="上次尝试失败",
                max_skills=3
            )
        """
        if not prompt_template:
            raise ValueError(
                "必须提供 prompt_template。"
                "框架不再提供默认模板，请用户根据具体场景自定义 prompt。"
            )
        
        # 格式化上下文
        context_str = "\n".join([
            f"- {k}: {str(v)}" for k, v in context.to_dict().items()
        ])
        
        # 格式化候选 Skill
        skills_str = ""
        for i, skill in enumerate(candidates, 1):
            cat_str = f"[{skill.category}]" if skill.category else "[?]"
            skills_str += f"\n{i}. {skill.name} {cat_str}\n"
            skills_str += f"   描述: {skill.description}\n"
            if skill.metadata:
                skills_str += f"   标签: {skill.metadata}\n"
        
        # 准备模板变量
        template_vars = {
            'context_str': context_str,
            'skills_str': skills_str,
            'candidates': candidates,
            'context': context,
        }
        # 将用户的额外参数合并进去
        template_vars.update(kwargs)
        
        try:
            return prompt_template.format(**template_vars)
        except KeyError as e:
            raise KeyError(
                f"模板变量 {e} 未提供。"
                f"可用的预定义变量: context_str, skills_str, candidates, context。"
                f"用户自定义变量通过 **kwargs 传入。"
            )
    
    def parse_llm_response(
        self,
        llm_response: str,
        candidates: List[SkillMetadata]
    ) -> List[SkillMetadata]:
        """
        解析 LLM 响应，提取选中的 Skill（参考 common_utils.py 的稳健解析）
        
        Args:
            llm_response: LLM 的响应文本
            candidates: 候选 Skill 列表
        
        Returns:
            选中的 Skill 列表
        """
        import json
        import re
        
        # 策略 1: 直接解析
        try:
            result = json.loads(llm_response)
            return self._extract_skills_from_result(result, candidates)
        except json.JSONDecodeError:
            logger.debug("直接解析失败，尝试提取 JSON 块")
        
        # 策略 2: 提取 ```json 代码块
        json_blocks = re.findall(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
        for block in reversed(json_blocks):
            try:
                result = json.loads(block.strip())
                return self._extract_skills_from_result(result, candidates)
            except json.JSONDecodeError:
                continue
        
        # 策略 3: 提取通用代码块
        code_blocks = re.findall(r'```\s*(.*?)\s*```', llm_response, re.DOTALL)
        for block in reversed(code_blocks):
            block = block.strip()
            if block.startswith('{') and block.endswith('}'):
                try:
                    result = json.loads(block)
                    return self._extract_skills_from_result(result, candidates)
                except json.JSONDecodeError:
                    continue
        
        # 策略 4: 提取花括号包围的内容
        json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
        matches = re.findall(json_pattern, llm_response, re.DOTALL)
        for match in reversed(matches):
            try:
                result = json.loads(match)
                if isinstance(result, dict) and 'selected' in result:
                    return self._extract_skills_from_result(result, candidates)
            except json.JSONDecodeError:
                continue
        
        # 所有策略都失败
        logger.error(f"无法从 LLM 响应中提取有效的 JSON: {llm_response[:200]}...")
        return []
    
    def _extract_skills_from_result(
        self,
        result: dict,
        candidates: List[SkillMetadata]
    ) -> List[SkillMetadata]:
        """从解析结果中提取 Skill"""
        selected_names = result.get("selected", [])
        reason = result.get("reason", "未提供理由")
        
        logger.info(f"LLM 选择理由: {reason}")
        
        # 根据名称查找 Skill
        selected_skills = []
        skill_dict = {s.name: s for s in candidates}
        
        for name in selected_names:
            if name in skill_dict:
                selected_skills.append(skill_dict[name])
                logger.info(f"选中 Skill: {name}")
            else:
                logger.warning(f"Skill 不在候选列表中: {name}")
        
        return selected_skills
    
    def select(
        self,
        all_skills: List[SkillMetadata],
        context: SelectionContext,
        llm_generate_func: Optional[Any] = None,
        llm_prompt: Optional[str] = None,
        prompt_template: Optional[str] = None,
        category: Optional[str] = None,
        **kwargs
    ) -> List[SkillMetadata]:
        """
        完整的两阶段选择流程
        
        Args:
            all_skills: 所有可用的 Skill
            context: 选择上下文
            llm_generate_func: LLM 生成函数（接受 prompt 返回文本）
            llm_prompt: 直接提供的 LLM prompt（可选）
                如果提供，将跳过 build_llm_prompt，直接使用此 prompt
            prompt_template: prompt 模板（可选）
                如果提供了 llm_generate_func 但未提供 llm_prompt，则必须提供此参数
            category: 指定只从某个分类筛选（可选）
            **kwargs: 额外参数，传递给 build_llm_prompt 或用户自定义
        
        Returns:
            最终选中的 Skill 列表
        
        Raises:
            ValueError: 如果提供了 llm_generate_func 但既未提供 llm_prompt 也未提供 prompt_template
        
        示例：
            # 方式 1: 只进行粗筛（不使用 LLM）
            selected = selector.select(all_skills, context)
            
            # 方式 2: 使用自定义模板
            template = "上下文: {context_str}\n候选: {skills_str}\n请选择..."
            selected = selector.select(
                all_skills, context, llm_func,
                prompt_template=template
            )
            
            # 方式 3: 直接提供完整 prompt
            my_prompt = "请从以下 Skills 中选择..."
            selected = selector.select(
                all_skills, context, llm_func,
                llm_prompt=my_prompt
            )
            
            # 方式 4: 传递额外参数
            template = "上下文: {context_str}\n历史: {history}\n候选: {skills_str}"
            selected = selector.select(
                all_skills, context, llm_func,
                prompt_template=template,
                history="上次失败"
            )
        """
        # 阶段 1：粗筛
        candidates = self.coarse_filter(all_skills, context, category)
        
        if not candidates:
            logger.warning("粗筛后无候选 Skill")
            return []
        
        # 如果没有提供 LLM 函数，直接返回粗筛结果
        if llm_generate_func is None:
            logger.info("未提供 LLM 函数，返回粗筛结果")
            return candidates
        
        # 阶段 2：LLM 精筛
        # 如果没有直接提供 prompt，则必须提供 template
        if llm_prompt is None:
            if prompt_template is None:
                raise ValueError(
                    "提供了 llm_generate_func 但未提供 llm_prompt 或 prompt_template。"
                    "请提供其中之一以进行 LLM 精筛。"
                )
            llm_prompt = self.build_llm_prompt(
                candidates, 
                context, 
                prompt_template,
                **kwargs
            )
        
        try:
            llm_response = llm_generate_func(llm_prompt)
            selected = self.parse_llm_response(llm_response, candidates)
            
            if not selected:
                logger.warning("LLM 未选中任何 Skill，返回粗筛结果")
                return candidates
            
            return selected
        
        except Exception as e:
            logger.error(f"LLM 精筛失败: {e}，返回粗筛结果")
            return candidates
    
    def get_children_skills(
        self,
        parent_skill: SkillMetadata,
        all_skills: List[SkillMetadata]
    ) -> List[SkillMetadata]:
        """
        获取父 Skill 的所有子 Skill
        
        Args:
            parent_skill: 父 Skill
            all_skills: 所有可用的 Skill
        
        Returns:
            子 Skill 列表
        """
        if not parent_skill.structure:
            return []
        
        child_names = parent_skill.structure.child_skills
        skill_dict = {s.name: s for s in all_skills}
        
        children = []
        for name in child_names:
            if name in skill_dict:
                children.append(skill_dict[name])
            else:
                logger.warning(f"子 Skill 不存在: {name} (父: {parent_skill.name})")
        
        return children


def build_prompt_with_skills(
    selected_skills: List[SkillMetadata],
    task_description: str,
    include_full_content: bool = True
) -> str:
    """
    使用选中的 Skill 构建最终的生成 prompt
    
    Args:
        selected_skills: 选中的 Skill 列表
        task_description: 任务描述
        include_full_content: 是否包含完整的 Skill 内容
    
    Returns:
        构建好的 prompt
    """
    skills_content = ""
    
    for skill in selected_skills:
        skills_content += f"\n# {skill.name}\n\n"
        if include_full_content:
            skills_content += skill.content
        else:
            skills_content += skill.description
        skills_content += "\n\n" + "="*60 + "\n"
    
    prompt = f"""# 相关知识和策略

{skills_content}

# 任务

{task_description}

请根据上述知识和策略完成任务。
"""
    
    return prompt
