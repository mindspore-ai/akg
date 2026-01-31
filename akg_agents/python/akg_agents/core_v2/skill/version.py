"""
Skill 版本管理

核心功能：
- 版本解析和比较（SemVer）
- 多版本管理
- 版本选择策略（最新/最旧/指定）

使用场景：
- 稳定仓库 + 测试仓库，灵活选择版本
- 支持同一 Skill 的多个版本共存
"""

import re
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import logging

from .metadata import SkillMetadata

logger = logging.getLogger(__name__)


@dataclass
class Version:
    """
    语义化版本（SemVer）
    
    格式：MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
    例如：1.2.3-alpha.1+build.123
    """
    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""
    
    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """解析版本字符串"""
        # SemVer 正则表达式
        pattern = r"^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$"
        match = re.match(pattern, version_str)
        
        if not match:
            raise ValueError(f"无效的版本格式: {version_str}")
        
        major, minor, patch, prerelease, build = match.groups()
        
        return cls(
            major=int(major),
            minor=int(minor),
            patch=int(patch),
            prerelease=prerelease or "",
            build=build or ""
        )
    
    def __str__(self) -> str:
        """转换为字符串"""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            version += f"-{self.prerelease}"
        if self.build:
            version += f"+{self.build}"
        return version
    
    def __lt__(self, other: "Version") -> bool:
        """小于比较"""
        # 比较主版本号
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        
        # 处理 prerelease（有 prerelease 的版本小于无 prerelease 的版本）
        if not self.prerelease and other.prerelease:
            return False
        if self.prerelease and not other.prerelease:
            return True
        if self.prerelease and other.prerelease:
            return self.prerelease < other.prerelease
        
        # 版本号完全相同
        return False
    
    def __eq__(self, other: object) -> bool:
        """等于比较"""
        if not isinstance(other, Version):
            return False
        # build 不参与版本比较
        return (
            self.major == other.major
            and self.minor == other.minor
            and self.patch == other.patch
            and self.prerelease == other.prerelease
        )
    
    def __le__(self, other: "Version") -> bool:
        """小于等于"""
        return self < other or self == other
    
    def __gt__(self, other: "Version") -> bool:
        """大于"""
        return not self <= other
    
    def __ge__(self, other: "Version") -> bool:
        """大于等于"""
        return not self < other
    
    def __hash__(self) -> int:
        """哈希值（不包含 build）"""
        return hash((self.major, self.minor, self.patch, self.prerelease))


VersionStrategy = Literal["latest", "oldest", "stable"]


class VersionManager:
    """
    版本管理器
    
    功能：
    - 注册多版本 Skill
    - 查询指定版本
    - 版本选择策略（最新/最旧/稳定）
    """
    
    def __init__(self):
        """初始化版本管理器"""
        # 按 skill_name -> version_str -> SkillMetadata
        self._skills: Dict[str, Dict[str, SkillMetadata]] = {}
    
    def register_skill(self, skill: SkillMetadata):
        """
        注册一个 Skill
        
        Args:
            skill: SkillMetadata 对象
        """
        if skill.name not in self._skills:
            self._skills[skill.name] = {}
        
        self._skills[skill.name][skill.version] = skill
        logger.debug(f"注册 Skill: {skill.name} v{skill.version}")
    
    def get_versions(self, skill_name: str) -> List[str]:
        """
        获取指定 Skill 的所有版本（按版本号排序）
        
        Args:
            skill_name: Skill 名称
        
        Returns:
            版本号列表（从旧到新）
        """
        if skill_name not in self._skills:
            return []
        
        versions = list(self._skills[skill_name].keys())
        
        # 尝试按 SemVer 排序
        try:
            versions.sort(key=lambda v: Version.parse(v))
        except ValueError as e:
            # 如果解析失败，降级到字符串排序
            logger.warning(f"版本排序失败（{skill_name}），使用字符串排序: {e}")
            versions.sort()
        
        return versions
    
    def get_latest_version(self, skill_name: str) -> Optional[str]:
        """
        获取最新版本号
        
        Args:
            skill_name: Skill 名称
        
        Returns:
            最新版本号，如果不存在返回 None
        """
        versions = self.get_versions(skill_name)
        return versions[-1] if versions else None
    
    def get_oldest_version(self, skill_name: str) -> Optional[str]:
        """
        获取最旧（稳定）版本号
        
        Args:
            skill_name: Skill 名称
        
        Returns:
            最旧版本号，如果不存在返回 None
        """
        versions = self.get_versions(skill_name)
        return versions[0] if versions else None
    
    def get_skill(
        self,
        skill_name: str,
        version: Optional[str] = None,
        strategy: VersionStrategy = "latest"
    ) -> Optional[SkillMetadata]:
        """
        获取指定版本的 Skill
        
        Args:
            skill_name: Skill 名称
            version: 指定版本号（如果提供，忽略 strategy）
            strategy: 版本选择策略
                - "latest": 最新版本（默认）
                - "oldest": 最旧版本（稳定版本）
                - "stable": 同 "oldest"
        
        Returns:
            SkillMetadata 对象，如果不存在返回 None
        
        Examples:
            >>> # 获取最新版本
            >>> manager.get_skill("cuda-basics")
            
            >>> # 获取稳定版本
            >>> manager.get_skill("cuda-basics", strategy="oldest")
            
            >>> # 获取指定版本
            >>> manager.get_skill("cuda-basics", version="1.0.0")
        """
        if skill_name not in self._skills:
            return None
        
        # 如果指定了版本，直接返回
        if version is not None:
            return self._skills[skill_name].get(version)
        
        # 根据策略选择版本
        if strategy == "latest":
            target_version = self.get_latest_version(skill_name)
        elif strategy in ("oldest", "stable"):
            target_version = self.get_oldest_version(skill_name)
        else:
            logger.warning(f"未知的版本策略: {strategy}，使用 'latest'")
            target_version = self.get_latest_version(skill_name)
        
        if target_version is None:
            return None
        
        return self._skills[skill_name].get(target_version)
    
    def get_all_skills(self, skill_name: str) -> List[SkillMetadata]:
        """
        获取指定 Skill 的所有版本
        
        Args:
            skill_name: Skill 名称
        
        Returns:
            SkillMetadata 列表（按版本号排序）
        """
        if skill_name not in self._skills:
            return []
        
        versions = self.get_versions(skill_name)
        return [self._skills[skill_name][v] for v in versions]
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        total_skills = len(self._skills)
        total_versions = sum(len(versions) for versions in self._skills.values())
        multi_version_skills = sum(1 for versions in self._skills.values() if len(versions) > 1)
        
        avg_versions = round(total_versions / total_skills, 2) if total_skills > 0 else 0
        
        return {
            "total_skills": total_skills,
            "total_versions": total_versions,
            "multi_version_skills": multi_version_skills,
            "avg_versions_per_skill": avg_versions,
        }


def compare_versions(v1: str, v2: str) -> int:
    """
    比较两个版本号
    
    Args:
        v1: 版本号 1
        v2: 版本号 2
    
    Returns:
        - 负数：v1 < v2
        - 0：v1 == v2
        - 正数：v1 > v2
    
    Examples:
        >>> compare_versions("1.0.0", "2.0.0")
        -1
        >>> compare_versions("2.0.0", "1.0.0")
        1
        >>> compare_versions("1.0.0", "1.0.0")
        0
    """
    try:
        version1 = Version.parse(v1)
        version2 = Version.parse(v2)
        
        if version1 < version2:
            return -1
        elif version1 > version2:
            return 1
        else:
            return 0
    except ValueError as e:
        logger.warning(f"版本比较失败，降级到字符串比较: {e}")
        # 降级到数字分割比较
        try:
            v1_parts = tuple(int(x) for x in v1.split('.'))
            v2_parts = tuple(int(x) for x in v2.split('.'))
            
            if v1_parts < v2_parts:
                return -1
            elif v1_parts > v2_parts:
                return 1
            else:
                return 0
        except:
            # 最后降级到字符串比较
            if v1 < v2:
                return -1
            elif v1 > v2:
                return 1
            else:
                return 0
