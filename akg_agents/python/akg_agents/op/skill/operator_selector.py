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
算子生成领域的 Skill 选择器

提供算子生成场景的专用 Skill 选择功能：
- OperatorSelectionContext: 算子生成上下文
- OperatorSkillSelector: 算子生成专用选择器
- 自动过滤器：backend, dsl, operator_type（用户无需手动创建）

推荐用法（简单）：
    from akg_agents.op.skill import (
        OperatorSkillSelector,
        OperatorSelectionContext
    )
    from akg_agents.core_v2.skill import SkillRegistry
    
    # 1. 加载 Skills
    registry = SkillRegistry()
    registry.load_from_directory(Path("~/.akg/skills"))
    
    # 2. 创建选择器（自动配置过滤器）
    selector = OperatorSkillSelector()
    
    # 3. 定义上下文（只需设置参数，自动过滤）
    context = OperatorSelectionContext(
        operator_type="softmax",
        dsl="triton",
        backend="cuda"
    )
    
    # 4. 选择 Skills
    # 方式 A：只进行粗筛（基于 metadata）
    selected = selector.coarse_filter(registry.get_all(), context)
    
    # 方式 B：两阶段筛选（粗筛 + LLM 精筛）
    selected = selector.select(registry.get_all(), context, llm_func)

高级用法（添加自定义过滤器）：
    # 如果需要额外的过滤条件
    def perf_filter(skill, context):
        return skill.metadata.get("performance") == "high"
    
    selector = OperatorSkillSelector(additional_filters=[perf_filter])
"""

from typing import List, Callable, Optional, Dict, Any
from dataclasses import dataclass

from akg_agents.core_v2.skill.skill_selector import (
    SelectionContext,
    SkillSelector,
    create_metadata_matcher,  # 从通用框架导入
    create_category_filter   # Category 筛选器
)
from akg_agents.core_v2.skill.metadata import SkillMetadata

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
        hardware: 硬件型号（如："ascend910b4", "a100"）
        
        继承自 SelectionContext 的 Category 筛选字段：
        include_categories: Category 白名单
        exclude_categories: Category 黑名单
        include_category_groups: 分组白名单（如 ["knowledge"]）
        exclude_category_groups: 分组黑名单
    
    示例：
        # 基本用法
        context = OperatorSelectionContext(
            operator_type="softmax",
            dsl="triton",
            backend="cuda",
            hardware="a100"
        )
        
        # 使用 Category 筛选：只要 guide 和 example 的 Skill
        context = OperatorSelectionContext(
            operator_type="softmax",
            dsl="triton-ascend",
            backend="ascend",
            include_categories=["guide", "example"]
        )
        
        # 使用分组筛选：只要 knowledge 组
        context = OperatorSelectionContext(
            dsl="triton-ascend",
            backend="ascend",
            include_category_groups=["knowledge"]
        )
        
        # 排除 workflow 和 agent
        context = OperatorSelectionContext(
            dsl="triton-ascend",
            backend="ascend",
            exclude_categories=["workflow", "agent"]
        )
        
        # 如果需要额外的字段，使用 custom_fields
        context = OperatorSelectionContext(
            operator_type="softmax",
            dsl="triton",
            backend="cuda",
            custom_fields={
                "optimization_goal": "speed",
                "task_type": "operator_generation"
            }
        )
    """
    # 算子生成特定字段
    operator_type: Optional[str] = None      # 算子类型
    dsl: Optional[str] = None                # DSL类型（triton, cuda等）
    backend: Optional[str] = None            # 后端（cuda, ascend等）
    hardware: Optional[str] = None           # 硬件型号（npu910b, a100等）
    framework: Optional[str] = None          # 框架（torch, mindspore等）


# ==================== 算子生成过滤器 ====================

# 使用通用的 create_metadata_matcher 工厂函数创建算子特定的过滤器
backend_filter = create_metadata_matcher("backend")
dsl_filter = create_metadata_matcher("dsl")
hardware_filter = create_metadata_matcher("hardware")
framework_filter = create_metadata_matcher("framework")
operator_type_filter = create_metadata_matcher("operator_type", "operator_patterns")

# Category 过滤器（支持 include 和 exclude 模式）
category_include_filter = create_category_filter("include")
category_exclude_filter = create_category_filter("exclude")


def create_operator_filters() -> List[Callable]:
    """
    创建算子生成领域的过滤器集合
    
    所有过滤器都是通过工厂函数生成的，避免代码重复。
    
    包含的过滤器：
    1. backend_filter: 后端匹配（cuda, ascend等）
    2. dsl_filter: DSL 匹配（triton, cuda等）
    3. hardware_filter: 硬件型号匹配（npu910b, a100等）
    4. framework_filter: 框架匹配（torch, mindspore等）
    5. operator_type_filter: 算子类型匹配（softmax, layernorm等）
    6. category_include_filter: Category 白名单（只包含指定 category）
    7. category_exclude_filter: Category 黑名单（排除指定 category）
    
    Returns:
        过滤器函数列表
    
    扩展说明：
        如果需要添加新的过滤维度（如 version 等），只需：
        1. 在 OperatorSelectionContext 中添加字段
        2. 在此函数中添加一行：xxx_filter = create_metadata_matcher("xxx")
        3. 将 xxx_filter 添加到返回列表
    
    示例：
        # 推荐用法：直接使用 OperatorSkillSelector（自动应用所有过滤器）
        selector = OperatorSkillSelector()
        context = OperatorSelectionContext(
            operator_type="softmax",
            dsl="triton",
            backend="cuda",
            hardware="a100"
        )
        candidates = selector.coarse_filter(all_skills, context)
        
        # 使用 Category 筛选：只要 guide, example 级别的 Skill
        context = OperatorSelectionContext(
            dsl="triton-ascend",
            backend="ascend",
            include_categories=["guide", "example"]
        )
        candidates = selector.coarse_filter(all_skills, context)
        
        # 使用分组筛选：排除 workflow 和 agent
        context = OperatorSelectionContext(
            dsl="triton-ascend",
            backend="ascend",
            exclude_category_groups=["orchestration", "actor"]
        )
        candidates = selector.coarse_filter(all_skills, context)
        
        # 高级用法 1：只使用部分过滤器
        from akg_agents.core_v2.skill import SkillSelector
        
        filters = [backend_filter, dsl_filter]
        custom_selector = SkillSelector(custom_filters=filters)
        
        # 高级用法 2：组合 include 和 exclude
        from akg_agents.core_v2.skill import create_metadata_matcher
        
        exclude_cpu = create_metadata_matcher("backend", "backend", "exclude")
        context = OperatorSelectionContext(backend="cpu")  # 要排除的后端
        filters = [dsl_filter, exclude_cpu]  # dsl 匹配 + 排除 cpu
        
        # 高级用法 3：使用逻辑组合器
        from akg_agents.core_v2.skill import and_filters
        
        # backend 必须是 cuda，且 dsl 不能是 opencl
        exclude_opencl = create_metadata_matcher("dsl", "dsl", "exclude")
        complex_filter = and_filters(backend_filter, exclude_opencl)
        custom_selector = SkillSelector(custom_filters=[complex_filter])
    """
    return [
        backend_filter, 
        dsl_filter, 
        hardware_filter, 
        framework_filter,
        operator_type_filter,
        category_include_filter,
        category_exclude_filter
    ]


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
    
    注意：
        此函数主要用于需要添加额外自定义过滤器的高级场景。
        对于常规使用，直接实例化 OperatorSkillSelector() 即可。
    
    Args:
        additional_filters: 额外的自定义过滤器（可选）
    
    Returns:
        配置好的 OperatorSkillSelector
    
    示例：
        # 推荐用法：直接使用类（99% 的场景）
        selector = OperatorSkillSelector()
        context = OperatorSelectionContext(
            operator_type="softmax",
            dsl="triton",
            backend="cuda"
        )
        selected = selector.coarse_filter(all_skills, context)
        
        # 高级用法：需要额外过滤器时（极少场景）
        def perf_filter(skill, context):
            return skill.metadata.get("performance") == "high"
        
        selector = create_operator_selector(additional_filters=[perf_filter])
        # 或者直接：
        # selector = OperatorSkillSelector(additional_filters=[perf_filter])
    """
    return OperatorSkillSelector(additional_filters=additional_filters)
