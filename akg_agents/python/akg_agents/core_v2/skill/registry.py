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
Skill注册表

管理已加载的Skill，提供加载、注册、查询、过滤、版本管理等功能
"""

from typing import Dict, List, Optional, Set, Literal
from pathlib import Path
import logging

from .metadata import SkillMetadata
from .loader import SkillLoader

logger = logging.getLogger(__name__)

# 版本选择策略类型
VersionStrategy = Literal["latest", "oldest", "stable"]


class SkillRegistry:
    """
    Skill注册表
    
    负责加载、管理和查询Skill
    
    功能：
    1. 从文件系统加载 Skills
    2. 注册和管理 Skills
    3. 查询和过滤 Skills
    4. 版本管理
    
    使用方式：
        # 方式1：直接加载（推荐）
        registry = SkillRegistry()
        registry.load_from_directory(Path("./skills"))
        
        # 方式2：加载多个目录
        registry = SkillRegistry()
        registry.load_from_directory(Path("./skills_stable"))
        registry.load_from_directory(Path("./skills_test"))
        
        # 方式3：手动注册（高级用法）
        skills = [...]  # 自定义加载逻辑
        registry.register_batch(skills)
    """
    
    def __init__(self, base_dir: Optional[Path] = None, enable_global: bool = False):
        """
        初始化注册表
        
        Args:
            base_dir: 项目根目录（用于自动发现 Skills）
            enable_global: 是否启用全局 Skill 目录（默认False，避免加载历史遗留skills）
        """
        # 内部 Loader（用于加载 Skills）
        self._loader = SkillLoader(base_dir=base_dir, enable_global=enable_global)
        
        # 按name索引的Skill字典
        self._skills_by_name: Dict[str, SkillMetadata] = {}
        
        # 按 category 索引的 Skill 集合
        self._skills_by_category: Dict[str, List[SkillMetadata]] = {}
        
        # 按版本索引（支持同名不同版本）
        self._skills_by_version: Dict[str, Dict[str, SkillMetadata]] = {}
    
    def register(self, skill: SkillMetadata, update_latest: bool = True) -> bool:
        """
        注册一个Skill（支持多版本）
        
        Args:
            skill: 要注册的SkillMetadata
            update_latest: 是否更新为最新版本（默认True）
        
        Returns:
            是否成功注册
        """
        # 验证Skill
        is_valid, error_msg = skill.validate()
        if not is_valid:
            logger.error(f"注册失败，Skill验证错误: {error_msg}")
            return False
        
        # 1. 添加到版本索引
        if skill.name not in self._skills_by_version:
            self._skills_by_version[skill.name] = {}
        
        # 检查版本是否已存在
        if skill.version in self._skills_by_version[skill.name]:
            logger.warning(f"版本已存在，覆盖: {skill.name} v{skill.version}")
        
        self._skills_by_version[skill.name][skill.version] = skill
        
        # 2. 更新主索引和 category 索引（只保留最新版本）
        current_latest = self._skills_by_name.get(skill.name)
        
        if update_latest:
            # 比较版本号，更新为最新
            if not current_latest or self._is_newer_version(skill.version, current_latest.version):
                # 移除旧的最新版本的 category 索引
                if current_latest and current_latest.category:
                    cat_list = self._skills_by_category.get(current_latest.category, [])
                    try:
                        cat_list.remove(current_latest)
                    except ValueError:
                        pass
                
                # 设置新的最新版本
                self._skills_by_name[skill.name] = skill
                
                # 添加到 category 索引
                if skill.category:
                    if skill.category not in self._skills_by_category:
                        self._skills_by_category[skill.category] = []
                    self._skills_by_category[skill.category].append(skill)
                
                logger.info(f"成功注册Skill并更新为最新版本: {skill}")
        else:
            # 不更新latest，但如果是第一个版本，还是要设置
            if not current_latest:
                self._skills_by_name[skill.name] = skill
                if skill.category:
                    if skill.category not in self._skills_by_category:
                        self._skills_by_category[skill.category] = []
                    self._skills_by_category[skill.category].append(skill)
                logger.info(f"成功注册首个Skill版本: {skill}")
            else:
                logger.info(f"成功注册Skill版本（不更新latest）: {skill}")
        
        return True
    
    def load_from_directory(self, skill_dir: Path, update_latest: bool = True) -> int:
        """
        从指定目录加载并注册 Skills
        
        Args:
            skill_dir: Skill 目录路径
            update_latest: 是否更新为最新版本（默认 True）
        
        Returns:
            成功注册的 Skill 数量
        
        示例：
            registry = SkillRegistry()
            count = registry.load_from_directory(Path("./skills"))
            print(f"加载了 {count} 个 Skills")
        """
        skills = self._loader.load_from_directory(skill_dir)
        return self.register_batch(skills, update_latest=update_latest)
    
    def load_all(self, additional_dirs: Optional[List[Path]] = None, update_latest: bool = True) -> int:
        """
        自动发现并加载所有标准路径的 Skills
        
        Args:
            additional_dirs: 额外的 Skill 目录列表
            update_latest: 是否更新为最新版本（默认 True）
        
        Returns:
            成功注册的 Skill 数量
        
        示例：
            registry = SkillRegistry()
            count = registry.load_all()
            print(f"加载了 {count} 个 Skills")
        """
        skills = self._loader.load_all(additional_dirs=additional_dirs)
        return self.register_batch(skills, update_latest=update_latest)
    
    def load_single(self, skill_path: Path, update_latest: bool = True) -> bool:
        """
        加载并注册单个 SKILL.md 文件
        
        Args:
            skill_path: SKILL.md 文件路径
            update_latest: 是否更新为最新版本（默认 True）
        
        Returns:
            是否成功注册
        
        示例：
            registry = SkillRegistry()
            success = registry.load_single(Path("./skills/cuda-basics/SKILL.md"))
        """
        skill = self._loader.load_single(skill_path)
        if skill:
            return self.register(skill, update_latest=update_latest)
        return False
    
    def _is_newer_version(self, v1: str, v2: str) -> bool:
        """比较版本号大小（v1 > v2返回True）"""
        try:
            # 使用 version.py 的 Version 类进行精确比较
            from .version import Version
            version1 = Version.parse(v1)
            version2 = Version.parse(v2)
            return version1 > version2
        except:
            # 如果解析失败，使用字符串比较
            logger.warning(f"版本解析失败，使用字符串比较: {v1} vs {v2}")
            return v1 > v2
    
    def register_batch(self, skills: List[SkillMetadata], update_latest: bool = True) -> int:
        """
        批量注册Skill
        
        Args:
            skills: 要注册的SkillMetadata列表
            update_latest: 是否更新为最新版本（默认 True）
        
        Returns:
            成功注册的数量
        """
        success_count = 0
        for skill in skills:
            if self.register(skill, update_latest=update_latest):
                success_count += 1
        
        logger.info(f"批量注册完成: 成功{success_count}/{len(skills)}")
        return success_count
    
    def get(
        self,
        name: str,
        version: Optional[str] = None,
        strategy: VersionStrategy = "latest"
    ) -> Optional[SkillMetadata]:
        """
        获取指定名称的Skill（支持版本选择策略）
        
        Args:
            name: Skill名称
            version: 指定版本号（如果提供，忽略 strategy）
            strategy: 版本选择策略（默认 "latest"）
                - "latest": 最新版本（默认）
                - "oldest": 最旧版本（稳定版本）
                - "stable": 同 "oldest"
        
        Returns:
            SkillMetadata实例，如果不存在返回None
        
        Examples:
            >>> # 获取最新版本（默认）
            >>> registry.get("cuda-basics")
            
            >>> # 获取稳定版本（最旧）
            >>> registry.get("cuda-basics", strategy="oldest")
            
            >>> # 获取指定版本
            >>> registry.get("cuda-basics", version="1.0.0")
        """
        # 如果指定了版本，直接返回
        if version is not None:
            versions = self._skills_by_version.get(name, {})
            return versions.get(version)
        
        # 如果没有这个 Skill，返回 None
        if name not in self._skills_by_version:
            return None
        
        # 根据策略选择版本
        if strategy == "latest":
            # 返回最新版本（已在 _skills_by_name 中维护）
            return self._skills_by_name.get(name)
        elif strategy in ("oldest", "stable"):
            # 返回最旧版本
            all_versions = self.get_all_versions(name)
            return all_versions[0] if all_versions else None
        else:
            logger.warning(f"未知的版本策略: {strategy}，使用 'latest'")
            return self._skills_by_name.get(name)
    
    def get_all_versions(self, name: str) -> List[SkillMetadata]:
        """
        获取某个Skill的所有版本
        
        Args:
            name: Skill名称
        
        Returns:
            按版本号排序的SkillMetadata列表（从旧到新）
        """
        versions_dict = self._skills_by_version.get(name, {})
        
        # 使用 version.py 的 Version 类进行排序
        try:
            from .version import Version
            version_tuples = [(v, Version.parse(v)) for v in versions_dict.keys()]
            version_tuples.sort(key=lambda x: x[1])
            sorted_versions = [v[0] for v in version_tuples]
        except:
            # 如果解析失败，使用字符串排序
            logger.warning(f"版本解析失败，使用字符串排序: {name}")
            sorted_versions = sorted(versions_dict.keys())
        
        return [versions_dict[v] for v in sorted_versions]
    
    def get_by_category(self, category: str) -> List[SkillMetadata]:
        """
        获取指定 category 的所有 Skill
        
        Args:
            category: 技能分类（如 workflow, agent, guide, example）
        
        Returns:
            该分类的 SkillMetadata 列表
        """
        return self._skills_by_category.get(category, []).copy()
    
    def get_all(self, strategy: VersionStrategy = "latest") -> List[SkillMetadata]:
        """
        获取所有已注册的Skill（支持版本选择策略）
        
        Args:
            strategy: 版本选择策略（默认 "latest"）
                - "latest": 每个 Skill 的最新版本
                - "oldest": 每个 Skill 的最旧版本（稳定版本）
                - "stable": 同 "oldest"
        
        Returns:
            SkillMetadata 列表
        
        Examples:
            >>> # 获取所有最新版本（默认）
            >>> registry.get_all()
            
            >>> # 获取所有稳定版本
            >>> registry.get_all(strategy="oldest")
        """
        if strategy == "latest":
            # 返回所有最新版本
            return list(self._skills_by_name.values())
        elif strategy in ("oldest", "stable"):
            # 返回所有最旧版本
            result = []
            for name in self._skills_by_version.keys():
                all_versions = self.get_all_versions(name)
                if all_versions:
                    result.append(all_versions[0])  # 最旧版本
            return result
        else:
            logger.warning(f"未知的版本策略: {strategy}，使用 'latest'")
            return list(self._skills_by_name.values())
    
    def get_names(self) -> Set[str]:
        """获取所有Skill名称"""
        return set(self._skills_by_name.keys())
    
    def get_versions(self, name: str) -> List[str]:
        """
        获取指定Skill的所有版本号（已排序）
        
        Args:
            name: Skill名称
        
        Returns:
            版本号列表（从旧到新）
        """
        versions_dict = self._skills_by_version.get(name, {})
        
        # 使用 version.py 的 Version 类进行排序
        try:
            from .version import Version
            version_tuples = [(v, Version.parse(v)) for v in versions_dict.keys()]
            version_tuples.sort(key=lambda x: x[1])
            return [v[0] for v in version_tuples]
        except:
            # 如果解析失败，使用字符串排序
            logger.warning(f"版本解析失败，使用字符串排序: {name}")
            return sorted(versions_dict.keys())
    
    def exists(self, name: str) -> bool:
        """
        检查Skill是否存在
        
        Args:
            name: Skill名称
        
        Returns:
            是否存在
        """
        return name in self._skills_by_name
    
    def unregister(self, name: str) -> bool:
        """
        注销Skill（移除所有相关索引）
        
        Args:
            name: Skill名称
        
        Returns:
            是否成功注销
        """
        if name not in self._skills_by_name:
            logger.warning(f"Skill不存在: {name}")
            return False
        
        # 从主索引获取skill
        skill = self._skills_by_name.pop(name)
        
        # 从所有索引中移除
        self._remove_from_indices(skill)
        
        logger.info(f"成功注销Skill: {name}")
        return True
    
    def clear(self):
        """清空注册表"""
        self._skills_by_name.clear()
        self._skills_by_category.clear()
        self._skills_by_version.clear()
        logger.info("注册表已清空")
    
    def filter(
        self,
        category: Optional[str] = None,
        name_pattern: Optional[str] = None,
        has_structure: Optional[bool] = None
    ) -> List[SkillMetadata]:
        """
        过滤Skill
        
        Args:
            category: 按分类过滤（如 workflow, agent, guide）
            name_pattern: 按名称模式过滤（支持通配符*）
            has_structure: 是否包含结构配置
        
        Returns:
            符合条件的SkillMetadata列表
        """
        # 起始集合
        if category:
            skills = self.get_by_category(category)
        else:
            skills = self.get_all()
        
        # 名称模式过滤
        if name_pattern:
            import fnmatch
            skills = [s for s in skills if fnmatch.fnmatch(s.name, name_pattern)]
        
        # 结构配置过滤
        if has_structure is not None:
            if has_structure:
                skills = [s for s in skills if s.structure is not None]
            else:
                skills = [s for s in skills if s.structure is None]
        
        return skills
    
    def get_statistics(self) -> Dict[str, any]:
        """
        获取注册表统计信息
        
        Returns:
            统计信息字典
        """
        # 计算总版本数
        total_versions = sum(len(versions) for versions in self._skills_by_version.values())
        
        # 计算多版本 Skill 数量
        multi_version_count = sum(
            1 for versions in self._skills_by_version.values()
            if len(versions) > 1
        )
        
        stats = {
            "total": len(self._skills_by_name),
            "total_versions": total_versions,
            "multi_version_skills": multi_version_count,
            "by_category": {
                cat: len(skills)
                for cat, skills in self._skills_by_category.items()
                if len(skills) > 0  # 只显示有 Skill 的分类
            },
            "with_structure": len([
                s for s in self._skills_by_name.values()
                if s.structure is not None
            ])
        }
        return stats
    
    def _remove_from_indices(self, skill: SkillMetadata):
        """从所有索引中移除Skill"""
        # 1. 从 category 索引移除
        if skill.category and skill.category in self._skills_by_category:
            try:
                self._skills_by_category[skill.category].remove(skill)
            except ValueError:
                pass
        
        # 2. 从version索引移除
        if skill.name in self._skills_by_version:
            self._skills_by_version[skill.name].pop(skill.version, None)
            # 如果该name下没有任何版本了，移除整个name entry
            if not self._skills_by_version[skill.name]:
                self._skills_by_version.pop(skill.name, None)
    
    def __len__(self) -> int:
        """返回注册的Skill数量"""
        return len(self._skills_by_name)
    
    def __contains__(self, name: str) -> bool:
        """检查Skill是否存在"""
        return name in self._skills_by_name
    
    def get_children(self, parent_name: str) -> List[SkillMetadata]:
        """
        获取子 Skill（基于 structure.child_skills）
        
        Args:
            parent_name: 父 Skill 名称
        
        Returns:
            子 Skill 列表
        """
        parent = self.get(parent_name)
        if not parent or not parent.structure:
            return []
        
        child_names = parent.structure.child_skills
        children = []
        for name in child_names:
            child = self.get(name)
            if child:
                children.append(child)
            else:
                logger.warning(f"子 Skill 不存在: {name} (父: {parent_name})")
        
        return children
    
    def get_default_children(self, parent_name: str) -> List[SkillMetadata]:
        """
        获取默认激活的子 Skill（基于 structure.default_children）
        
        Args:
            parent_name: 父 Skill 名称
        
        Returns:
            默认子 Skill 列表
        """
        parent = self.get(parent_name)
        if not parent or not parent.structure:
            return []
        
        default_names = parent.structure.default_children
        defaults = []
        for name in default_names:
            child = self.get(name)
            if child:
                defaults.append(child)
            else:
                logger.warning(f"默认子 Skill 不存在: {name} (父: {parent_name})")
        
        return defaults
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        cat_stats = " ".join([
            f"{cat}={count}"
            for cat, count in sorted(stats['by_category'].items())
            if count > 0  # 只显示有Skill的分类
        ])
        return f"<SkillRegistry total={stats['total']} {cat_stats}>"

