"""
算子生成领域的 Skill 选择器

扩展通用的 SkillSelector，提供算子生成领域特定的：
- OperatorSelectionContext: 算子生成上下文
- OperatorSkillSelector: 算子生成专用选择器
- 算子相关的过滤器（backend, dsl, operator_type）

示例：
    from skill_system.operator_selector import (
        OperatorSkillSelector,
        OperatorSelectionContext
    )
    
    # 创建算子选择器
    selector = OperatorSkillSelector()
    
    # 算子生成上下文
    context = OperatorSelectionContext(
        task_type="operator_generation",
        operator_type="softmax",
        dsl="triton",
        backend="cuda"
    )
    
    # 选择 Skills
    selected = selector.select(all_skills, context, llm_func)
"""

from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass

from ai_kernel_generator.core_v2.skill.skill_selector import SelectionContext, SkillSelector
from ai_kernel_generator.core_v2.skill.metadata import SkillMetadata

import logging

logger = logging.getLogger(__name__)


@dataclass
class OperatorSelectionContext(SelectionContext):
    """算子生成领域的专用选择上下文
    
    扩展通用上下文，添加算子生成特定的字段。
    
    字段说明：
        operator_type: 算子类型（如："softmax", "layernorm", "reduce"）
        dsl: DSL 类型（如："triton", "cuda", "opencl"）
        backend: 后端（如："cuda", "ascend", "rocm"）
    
    示例：
        context = OperatorSelectionContext(
            task_type="operator_generation",
            operator_type="softmax",
            dsl="triton",
            backend="cuda",
            optimization_goal="speed"
        )
    """
    # 算子生成特定字段
    operator_type: Optional[str] = None      # 算子类型
    dsl: Optional[str] = None                # DSL类型（triton, cuda等）
    backend: Optional[str] = None            # 后端（cuda, ascend等）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = super().to_dict()
        result.update({
            "operator_type": self.operator_type,
            "dsl": self.dsl,
            "backend": self.backend
        })
        return {k: v for k, v in result.items() if v is not None}


# ==================== 算子生成过滤器 ====================


def backend_filter(skill: SkillMetadata, context: SelectionContext) -> bool:
    """检查 backend 匹配
    
    从 Skill 的 metadata 中提取 backend 字段，与 context 中的 backend 比较。
    
    Skill metadata 格式：
        metadata:
          backend: "cuda, ascend"  # 支持多个后端，逗号分隔
    
    Args:
        skill: 待过滤的 Skill
        context: 选择上下文
    
    Returns:
        是否通过过滤
    """
    # 只对 OperatorSelectionContext 进行过滤
    if not isinstance(context, OperatorSelectionContext):
        return True
    
    # 如果 context 没有指定 backend，放行
    if not context.backend:
        return True
    
    # 检查 Skill 的 backend metadata
    if skill.metadata:
        skill_backends = skill.metadata.get("backend", "")
        if skill_backends:
            backends = [b.strip() for b in skill_backends.split(",")]
            return context.backend in backends
    
    # 如果 Skill 没有 backend metadata，默认放行（可能是通用 Skill）
    return True


def operator_type_filter(skill: SkillMetadata, context: SelectionContext) -> bool:
    """检查 operator_type 匹配
    
    从 Skill 的 metadata 中提取 operator_patterns 字段，与 context 中的 operator_type 比较。
    
    Skill metadata 格式：
        metadata:
          operator_patterns: "softmax, layernorm, normalization"  # 支持多个模式
    
    Args:
        skill: 待过滤的 Skill
        context: 选择上下文
    
    Returns:
        是否通过过滤
    """
    if not isinstance(context, OperatorSelectionContext):
        return True
    
    if not context.operator_type:
        return True
    
    if skill.metadata:
        patterns_str = skill.metadata.get("operator_patterns", "")
        if patterns_str:
            patterns = [p.strip() for p in patterns_str.split(",")]
            return context.operator_type in patterns
    
    return True


def dsl_filter(skill: SkillMetadata, context: SelectionContext) -> bool:
    """检查 dsl 匹配
    
    从 Skill 的 metadata 中提取 dsl 字段，与 context 中的 dsl 比较。
    
    Skill metadata 格式：
        metadata:
          dsl: "triton"  # 或 "cuda", "opencl" 等
    
    Args:
        skill: 待过滤的 Skill
        context: 选择上下文
    
    Returns:
        是否通过过滤
    """
    if not isinstance(context, OperatorSelectionContext):
        return True
    
    if not context.dsl:
        return True
    
    if skill.metadata:
        skill_dsl = skill.metadata.get("dsl", "")
        if skill_dsl:
            dsls = [d.strip() for d in skill_dsl.split(",")]
            return context.dsl in dsls
    
    return True


def create_operator_filters() -> List[Callable]:
    """
    创建算子生成领域的过滤器集合
    
    包含的过滤器：
    1. backend_filter: 后端匹配（cuda, ascend等）
    2. operator_type_filter: 算子类型匹配（softmax, layernorm等）
    3. dsl_filter: DSL 匹配（triton, cuda等）
    
    Returns:
        过滤器函数列表
    
    示例：
        filters = create_operator_filters()
        selector = SkillSelector(custom_filters=filters)
        
        context = OperatorSelectionContext(
            operator_type="softmax",
            dsl="triton",
            backend="cuda"
        )
        candidates = selector.coarse_filter(skills, context)
    """
    return [backend_filter, operator_type_filter, dsl_filter]


# ==================== 算子生成选择器 ====================


class OperatorSkillSelector(SkillSelector):
    """
    算子生成专用的 Skill 选择器
    
    继承通用 SkillSelector，自动配置算子生成相关的过滤器。
    
    特点：
    - 自动加载 backend、operator_type、dsl 过滤器
    - 支持 OperatorSelectionContext
    - 可以额外添加自定义过滤器
    
    示例：
        # 基本使用
        selector = OperatorSkillSelector()
        context = OperatorSelectionContext(
            operator_type="softmax",
            dsl="triton",
            backend="cuda"
        )
        selected = selector.select(all_skills, context, llm_func)
        
        # 添加自定义过滤器
        def custom_filter(skill, context):
            return skill.metadata.get("performance") == "high"
        
        selector = OperatorSkillSelector(additional_filters=[custom_filter])
    """
    
    def __init__(self, additional_filters: Optional[List[Callable]] = None):
        """
        初始化算子生成选择器
        
        Args:
            additional_filters: 额外的自定义过滤器（在算子过滤器之后应用）
        """
        # 创建算子过滤器
        operator_filters = create_operator_filters()
        
        # 合并额外的过滤器
        if additional_filters:
            all_filters = operator_filters + additional_filters
        else:
            all_filters = operator_filters
        
        # 调用父类初始化
        super().__init__(custom_filters=all_filters)
        
        logger.info(f"初始化 OperatorSkillSelector，共 {len(all_filters)} 个过滤器")


# 便捷函数：快速创建算子选择器
def create_operator_selector(additional_filters: Optional[List[Callable]] = None) -> OperatorSkillSelector:
    """
    快速创建算子生成专用的 SkillSelector
    
    Args:
        additional_filters: 额外的自定义过滤器
    
    Returns:
        配置好的 OperatorSkillSelector
    
    示例：
        # 方式 1: 使用类
        selector = OperatorSkillSelector()
        
        # 方式 2: 使用便捷函数（等价）
        selector = create_operator_selector()
        
        # 方式 3: 添加自定义过滤器
        def perf_filter(skill, context):
            return skill.metadata.get("performance") == "high"
        
        selector = create_operator_selector(additional_filters=[perf_filter])
    """
    return OperatorSkillSelector(additional_filters=additional_filters)
