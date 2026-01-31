"""
Skill元数据定义与解析

支持标准SKILL.md格式：
---
name: skill-name
description: "技能描述"
level: L1
version: "1.0.0"
---
# Markdown内容
"""

import re
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path


class SkillLevel(str, Enum):
    """技能层级（通用定义，提供语义提示）"""
    L1 = "L1"  # 流程/编排层（Process/Orchestration）
    L2 = "L2"  # 组件/执行层（Component/Actor）
    L3 = "L3"  # 方法/策略层（Method/Strategy）
    L4 = "L4"  # 实现/细节层（Implementation/Detail）
    L5 = "L5"  # 原子/样例层（Atomic/Example）
    
    @classmethod
    def get_semantic_hint(cls, level: "SkillLevel") -> str:
        """获取语义提示"""
        hints = {
            cls.L1: "流程/编排层（Process/Orchestration）",
            cls.L2: "组件/执行层（Component/Actor）",
            cls.L3: "方法/策略层（Method/Strategy）",
            cls.L4: "实现/细节层（Implementation/Detail）",
            cls.L5: "原子/样例层（Atomic/Example）",
        }
        return hints.get(level, "未知层级")


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
    
    符合Agent Skills开放标准，并扩展支持AIKG分级管理
    """
    # ===== 必需字段（Agent Skills标准） =====
    name: str                          # 技能名称（小写字母数字+连字符）
    description: str                   # 技能描述（1-1024字符）
    
    # ===== 可选字段（Agent Skills标准） =====
    license: Optional[str] = None      # 许可证（MIT, Apache-2.0等）
    compatibility: Optional[str] = None # 兼容性标记
    metadata: Dict[str, str] = field(default_factory=dict)  # 自定义元数据
    
    # ===== AIKG扩展字段 =====
    level: Optional[SkillLevel] = None  # 技能层级（L1-L5）
    category: Optional[str] = None      # 语义类型（如：workflow, agent, strategy等）
    version: str = "1.0.0"              # 版本号
    structure: Optional[SkillStructure] = None  # 结构配置（层级关系）
    
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
        
        # 3. 验证level（如果指定）
        if self.level and self.level not in SkillLevel:
            return False, f"level必须是{[l.value for l in SkillLevel]}之一"
        
        
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
        
        # 提取AIKG扩展字段
        level_str = yaml_data.get("level")
        level = SkillLevel(level_str) if level_str else None
        category = yaml_data.get("category")
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
        
        return cls(
            name=name,
            description=description,
            license=license_val,
            compatibility=compatibility,
            metadata=custom_metadata,
            level=level,
            category=category,
            version=version,
            structure=structure,
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
        if self.level:
            result["level"] = self.level.value
        if self.category:
            result["category"] = self.category
        result["version"] = self.version
        
        if self.structure:
            result["structure"] = {
                "child_skills": self.structure.child_skills,
                "default_children": self.structure.default_children,
                "exclusive_groups": self.structure.exclusive_groups,
            }
        
        if self.skill_path:
            result["skill_path"] = str(self.skill_path)
        
        return result
    
    def __repr__(self) -> str:
        level_str = f"[{self.level.value}]" if self.level else ""
        return f"<SkillMetadata {level_str} {self.name} v{self.version}>"

