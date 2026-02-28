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
Skill元数据定义与解析

支持标准SKILL.md格式：
---
name: skill-name
description: "技能描述"
category: workflow
version: "1.0.0"
---
# Markdown内容
"""

import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# 标准 category 值（非枚举，仅作参考约束）
STANDARD_CATEGORIES = {
    "workflow": "工作流/编排（Process/Orchestration）",
    "overview": "系统概览（Overview）",
    "agent": "智能体/执行组件（Agent/Actor）",
    "guide": "设计方法与编程指南（Design & Programming Guide）",
    "fundamental": "基础知识（Fundamental）",
    "method": "优化方法/策略（Method/Strategy）",
    "implementation": "实现细节（Implementation）",
    "reference": "参考文档（Reference）",
    "example": "代码示例（Example）",
    "case": "具体案例（Case Study）",
}

# 粗粒度分组，用于 include_category_groups / exclude_category_groups 过滤
CATEGORY_GROUPS = {
    "orchestration": ["workflow", "overview"],
    "actor": ["agent"],
    "knowledge": ["guide", "fundamental", "method", "implementation", "reference"],
    "example": ["example", "case"],
}

# 旧 level 字段到默认 category 的映射（兼容未更新的 SKILL.md）
LEVEL_TO_DEFAULT_CATEGORY = {
    "L1": "workflow",
    "L2": "agent",
    "L3": "guide",
    "L4": "implementation",
    "L5": "example",
}


@dataclass
class SkillStructure:
    """技能结构配置（层级关系和约束）"""
    # 声明的子 Skill（下一层级）
    child_skills: List[str] = field(default_factory=list)
    
    # 默认激活的子 Skill
    default_children: List[str] = field(default_factory=list)
    
    # 互斥组（同组内 Skill 不能同时激活）
    exclusive_groups: List[List[str]] = field(default_factory=list)


@dataclass
class SkillMetadata:
    """
    Skill元数据
    
    符合Agent Skills开放标准，并扩展支持AIKG分类管理（category）
    """
    # ===== 必需字段（Agent Skills标准） =====
    name: str                          # 技能名称（小写字母数字+连字符）
    description: str                   # 技能描述（1-1024字符）
    
    # ===== 可选字段（Agent Skills标准） =====
    license: Optional[str] = None      # 许可证（MIT, Apache-2.0等）
    compatibility: Optional[str] = None # 兼容性标记
    metadata: Dict[str, str] = field(default_factory=dict)  # 自定义元数据
    
    # ===== AIKG扩展字段 =====
    category: Optional[str] = None      # 语义类型（如：workflow, agent, guide, example等）
    version: str = "1.0.0"              # 版本号
    structure: Optional[SkillStructure] = None  # 结构配置（层级关系）
    
    # ===== 工具联动字段 =====
    recommended_tools: List[str] = field(default_factory=list)  # 建议使用的工具列表
    tool_guidance: str = ""             # 工具使用指导（注入 prompt）
    
    # ===== 运行时字段 =====
    skill_path: Optional[Path] = None   # SKILL.md文件路径
    content: str = ""                   # Markdown内容
    
    # ===== 验证相关 =====
    _NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
    _MAX_NAME_LENGTH = 64
    _MIN_DESC_LENGTH = 1
    
    def validate(self) -> tuple[bool, Optional[str]]:
        """
        验证元数据是否符合规范
        
        Returns:
            (is_valid, error_message)
        """
        # 1. 验证name
        if not self.name:
            return False, "name字段不能为空"
        
        if len(self.name) > self._MAX_NAME_LENGTH:
            return False, f"name长度不能超过{self._MAX_NAME_LENGTH}字符"
        
        if not self._NAME_PATTERN.match(self.name):
            return False, "name必须是小写字母数字+单个连字符，不能以-开头/结尾"
        
        # 2. 验证description
        if not self.description:
            return False, "description字段不能为空"
        
        desc_len = len(self.description)
        if desc_len < self._MIN_DESC_LENGTH:
            return False, f"description长度必须至少{self._MIN_DESC_LENGTH}字符"
        
        # 3. 验证 category（如果指定，可选检查是否在标准列表中）
        if self.category and self.category not in STANDARD_CATEGORIES:
            logger.debug(f"category '{self.category}' 不在 STANDARD_CATEGORIES 中，将作为自定义 category 使用")
        
        return True, None
    
    @classmethod
    def from_yaml_dict(cls, yaml_data: Dict[str, Any], skill_path: Optional[Path] = None) -> "SkillMetadata":
        """
        从YAML字典创建元数据
        
        Args:
            yaml_data: YAML frontmatter解析后的字典
            skill_path: SKILL.md文件路径
        
        Returns:
            SkillMetadata实例
        """
        # 提取必需字段
        name = yaml_data.get("name", "")
        description = yaml_data.get("description", "")
        
        # 提取可选字段
        license_val = yaml_data.get("license")
        compatibility = yaml_data.get("compatibility")
        custom_metadata = yaml_data.get("metadata", {})
        
        # 提取AIKG扩展字段（兼容旧 level 字段：无 category 时有 level 则自动转换）
        category = yaml_data.get("category")
        level_str = yaml_data.get("level")
        if not category and level_str and level_str in LEVEL_TO_DEFAULT_CATEGORY:
            category = LEVEL_TO_DEFAULT_CATEGORY[level_str]
            logger.warning(
                f"SKILL.md 中使用了已废弃的 'level: {level_str}'，已自动转换为 category: {category}。"
                "请更新 frontmatter 为 category。"
            )
        version = yaml_data.get("version", "1.0.0")
        
        # 解析结构配置
        structure = None
        if "structure" in yaml_data:
            struct_data = yaml_data["structure"]
            structure = SkillStructure(
                child_skills=struct_data.get("child_skills", []),
                default_children=struct_data.get("default_children", []),
                exclusive_groups=struct_data.get("exclusive_groups", []),
            )
        
        # 解析工具联动字段
        recommended_tools = yaml_data.get("recommended_tools", [])
        tool_guidance = yaml_data.get("tool_guidance", "")
        
        return cls(
            name=name,
            description=description,
            license=license_val,
            compatibility=compatibility,
            metadata=custom_metadata,
            category=category,
            version=version,
            structure=structure,
            recommended_tools=recommended_tools,
            tool_guidance=tool_guidance,
            skill_path=skill_path
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = {
            "name": self.name,
            "description": self.description,
        }
        
        # 添加可选字段
        if self.license:
            result["license"] = self.license
        if self.compatibility:
            result["compatibility"] = self.compatibility
        if self.metadata:
            result["metadata"] = self.metadata
        
        # 添加AIKG扩展字段
        if self.category:
            result["category"] = self.category
        result["version"] = self.version
        
        if self.structure:
            result["structure"] = {
                "child_skills": self.structure.child_skills,
                "default_children": self.structure.default_children,
                "exclusive_groups": self.structure.exclusive_groups,
            }
        
        # 工具联动字段
        if self.recommended_tools:
            result["recommended_tools"] = self.recommended_tools
        if self.tool_guidance:
            result["tool_guidance"] = self.tool_guidance
        
        if self.skill_path:
            result["skill_path"] = str(self.skill_path)
        
        return result
    
    def __repr__(self) -> str:
        cat_str = f"[{self.category}]" if self.category else ""
        return f"<SkillMetadata {cat_str} {self.name} v{self.version}>"

