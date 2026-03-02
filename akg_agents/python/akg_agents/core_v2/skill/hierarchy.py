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
Skill 层级管理

提供层级关系查询和互斥检查功能。

注意：验证功能（循环依赖检查、层级合法性检查）已移至文件末尾，
     当前未启用，保留作为可选功能。
"""

from typing import Dict, List, Set, Optional, Tuple
import logging

from .metadata import SkillMetadata, CATEGORY_GROUPS
from .registry import SkillRegistry

# category 到层级序号的映射（用于可选验证：父 category 顺序 < 子 category 顺序）
_CATEGORY_ORDER = {
    cat: idx
    for idx, cats in enumerate(CATEGORY_GROUPS.values())
    for cat in cats
}

logger = logging.getLogger(__name__)


class SkillHierarchy:
    """
    Skill 层级管理器
    
    核心功能：
    1. 提供父子关系查询
    2. 提供互斥组冲突检查
    
    注意：验证功能（循环依赖、层级合法性）当前未启用。
    """
    
    def __init__(self, registry: SkillRegistry):
        """
        初始化层级管理器
        
        Args:
            registry: Skill 注册表
        """
        self.registry = registry
        self._parent_child_map: Dict[str, List[str]] = {}
        self._build_relations()
    
    def _build_relations(self):
        """构建父子关系映射"""
        self._parent_child_map.clear()
        
        for skill in self.registry.get_all():
            if skill.structure and skill.structure.child_skills:
                self._parent_child_map[skill.name] = skill.structure.child_skills
    
    def rebuild(self):
        """重新构建关系（当注册表更新后调用）"""
        self._build_relations()
    
    # ==================== 查询功能 ====================
    
    def get_children(self, parent_name: str) -> List[str]:
        """
        获取子 Skill 名称列表
        
        Args:
            parent_name: 父 Skill 名称
        
        Returns:
            子 Skill 名称列表
        """
        return self._parent_child_map.get(parent_name, []).copy()
    
    def get_parents(self, child_name: str) -> List[str]:
        """
        获取父 Skill 名称列表
        
        Args:
            child_name: 子 Skill 名称
        
        Returns:
            父 Skill 名称列表
        """
        parents = []
        for parent, children in self._parent_child_map.items():
            if child_name in children:
                parents.append(parent)
        return parents
    
    def get_descendants(self, skill_name: str, max_depth: int = 10) -> Set[str]:
        """
        获取所有后代 Skill（递归）
        
        Args:
            skill_name: Skill 名称
            max_depth: 最大递归深度
        
        Returns:
            所有后代 Skill 名称集合
        """
        descendants = set()
        
        def traverse(name: str, depth: int):
            if depth >= max_depth:
                return
            
            children = self._parent_child_map.get(name, [])
            for child in children:
                if child not in descendants:
                    descendants.add(child)
                    traverse(child, depth + 1)
        
        traverse(skill_name, 0)
        return descendants
    
    def check_exclusive_conflict(
        self,
        skill: SkillMetadata,
        active_skills: Set[str]
    ) -> Optional[str]:
        """
        检查互斥组冲突
        
        Args:
            skill: 要检查的 Skill
            active_skills: 当前激活的 Skill 名称集合
        
        Returns:
            如果有冲突，返回错误消息；否则返回 None
        """
        if not skill.structure or not skill.structure.exclusive_groups:
            return None
        
        for group in skill.structure.exclusive_groups:
            # 检查该组中有多少个已激活
            active_in_group = [name for name in group if name in active_skills]
            
            # 如果组内已有激活的，且要添加的也在组内，则冲突
            if active_in_group and skill.name in group:
                return f"互斥冲突: '{skill.name}' 与 {active_in_group} 不能同时激活"
        
        return None
    
    def __repr__(self) -> str:
        return (
            f"<SkillHierarchy "
            f"skills={len(self.registry)} "
            f"relations={len(self._parent_child_map)}>"
        )


# ============================================================================
# 验证功能（当前未启用，保留作为可选功能）
# ============================================================================
#
# 以下功能用于验证 Skill 层级关系的合法性，包括：
# 1. 循环依赖检测
# 2. 层级合法性检查（父层级 < 子层级）
# 3. 渐进式披露检查（父层级 + 1 = 子层级）
#
# 这些功能对于大多数使用场景不是必需的，因此暂时未启用。
# 如需使用，可以取消注释并调用 validate_all() 方法。
#
# 使用方式：
#     hierarchy = SkillHierarchy(registry)
#     is_valid, errors = hierarchy.validate_all()
#     if not is_valid:
#         for error in errors:
#             print(f"验证错误: {error}")
# ============================================================================


def _get_category_order_value(category: Optional[str]) -> int:
    """获取 category 对应的层级序号（用于父子顺序校验，越小越“上层”）"""
    if not category:
        return 999
    return _CATEGORY_ORDER.get(category, 999)


def validate_all(hierarchy: "SkillHierarchy") -> Tuple[bool, List[str]]:
    """
    验证所有 Skill 的层级关系（可选功能）
    
    注意：此功能当前未在系统中启用，仅保留作为可选的验证工具。
    
    Args:
        hierarchy: SkillHierarchy 实例
    
    Returns:
        (是否全部合法, 错误消息列表)
    
    检查项：
    1. 循环依赖检测
    2. 层级合法性（父层级 < 子层级）
    3. 渐进式披露（父层级 + 1 = 子层级）
    4. 子 Skill 存在性（仅警告）
    """
    errors = []
    
    # 1. 检查循环依赖
    cycles = detect_cycles(hierarchy)
    if cycles:
        errors.append(f"检测到循环依赖: {cycles}")
    
    # 2. 检查分类顺序合法性（父 category 顺序 < 子 category 顺序）
    for skill in hierarchy.registry.get_all():
        if skill.structure and skill.structure.child_skills:
            level_errors = _validate_category_order(hierarchy, skill)
            errors.extend(level_errors)
    
    # 3. 检查渐进式披露（父 category 顺序 + 1 = 子 category 顺序）
    for skill in hierarchy.registry.get_all():
        if skill.structure and skill.structure.child_skills:
            progressive_errors = _validate_progressive_disclosure(hierarchy, skill)
            errors.extend(progressive_errors)
    
    # 4. 检查子 Skill 是否存在（仅警告，不强制要求所有子 Skill 都加载）
    for skill in hierarchy.registry.get_all():
        if skill.structure and skill.structure.child_skills:
            for child_name in skill.structure.child_skills:
                if not hierarchy.registry.exists(child_name):
                    logger.warning(
                        f"Skill '{skill.name}' 声明的子 Skill '{child_name}' 未加载 "
                        f"(这不是错误，仅当实际需要使用时才需要加载)"
                    )
    
    return len(errors) == 0, errors


def detect_cycles(hierarchy: "SkillHierarchy") -> List[List[str]]:
    """
    检测循环依赖（可选功能）
    
    使用 DFS 算法检测依赖图中的环。
    
    Args:
        hierarchy: SkillHierarchy 实例
    
    Returns:
        检测到的循环路径列表
    """
    cycles = []
    visited = set()
    rec_stack = set()
    
    def dfs(node: str, path: List[str]):
        if node in rec_stack:
            # 找到循环
            cycle_start = path.index(node)
            cycles.append(path[cycle_start:] + [node])
            return
        
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        # 访问子节点
        children = hierarchy._parent_child_map.get(node, [])
        for child in children:
            dfs(child, path[:])
        
        rec_stack.remove(node)
    
    # 从所有节点开始 DFS
    for skill_name in hierarchy._parent_child_map.keys():
        if skill_name not in visited:
            dfs(skill_name, [])
    
    return cycles


def _validate_category_order(hierarchy: "SkillHierarchy", parent: SkillMetadata) -> List[str]:
    """
    验证父子分类顺序（父 category 顺序必须 < 子 category 顺序）（可选功能）
    
    Args:
        hierarchy: SkillHierarchy 实例
        parent: 父 Skill
    
    Returns:
        错误消息列表
    """
    errors = []
    
    if not parent.category or not parent.structure:
        return errors
    
    parent_order = _get_category_order_value(parent.category)
    
    for child_name in parent.structure.child_skills:
        child = hierarchy.registry.get(child_name)
        if not child or not child.category:
            continue
        
        child_order = _get_category_order_value(child.category)
        
        # 父顺序必须 < 子顺序
        if parent_order >= child_order:
            errors.append(
                f"分类顺序错误: '{parent.name}' (category={parent.category}) "
                f"不能指向更高或相同层级的 '{child_name}' (category={child.category})"
            )
    
    return errors


def _validate_progressive_disclosure(hierarchy: "SkillHierarchy", parent: SkillMetadata) -> List[str]:
    """
    验证渐进式披露（父 category 顺序 + 1 = 子 category 顺序）（可选功能）
    
    Args:
        hierarchy: SkillHierarchy 实例
        parent: 父 Skill
    
    Returns:
        错误消息列表
    """
    errors = []
    
    if not parent.category or not parent.structure:
        return errors
    
    parent_order = _get_category_order_value(parent.category)
    
    for child_name in parent.structure.child_skills:
        child = hierarchy.registry.get(child_name)
        if not child or not child.category:
            continue
        
        child_order = _get_category_order_value(child.category)
        
        # 父顺序 + 1 必须等于子顺序（渐进式）
        if child_order != parent_order + 1:
            errors.append(
                f"渐进式披露错误: '{parent.name}' (category={parent.category}) "
                f"只能引用下一层级，但引用了 '{child_name}' (category={child.category})"
            )
    
    return errors
