"""
Skill加载器

支持从多个路径加载SKILL.md文件：
- 项目级：.akg/skills, .claude/skills/, .opencode/skill/
- 全局级：~/.akg/skills/, ~/.claude/skills/, ~/.config/opencode/skill/
- 自定义路径
"""

import re
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Set
import logging

from .metadata import SkillMetadata

logger = logging.getLogger(__name__)


class SkillLoadError(Exception):
    """Skill加载错误"""
    pass


class SkillLoader:
    """
    Skill加载器
    
    负责从指定目录扫描和加载SKILL.md文件
    """
    
    # YAML frontmatter模式
    FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
    
    # 标准Skill目录名称
    STANDARD_PATHS = [
        ".akg/skills",
        ".claude/skills",
        ".opencode/skill",
    ]
    
    # 全局Skill目录
    GLOBAL_PATHS = [
        Path.home() / ".akg" / "skills",
        Path.home() / ".claude" / "skills",
        Path.home() / ".config" / "opencode" / "skill",
    ]
    
    def __init__(self, base_dir: Optional[Path] = None, enable_global: bool = False):
        """
        初始化Skill加载器
        
        Args:
            base_dir: 项目根目录（用于查找项目级Skill）
            enable_global: 是否启用全局Skill目录（默认False，避免加载历史遗留skills）
        """
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.enable_global = enable_global
        
        # 已加载的Skill路径（避免重复加载）
        self._loaded_paths: Set[Path] = set()
    
    def discover_skill_directories(self) -> List[Path]:
        """
        发现所有可能的Skill目录
        
        Returns:
            按优先级排序的Skill目录列表
        """
        directories = []
        
        # 1. 项目级目录（从当前目录向上查找到git根目录）
        current = self.base_dir.resolve()
        visited = set()
        
        while current not in visited:
            visited.add(current)
            
            # 检查标准路径
            for std_path in self.STANDARD_PATHS:
                skill_dir = current / std_path
                if skill_dir.exists() and skill_dir.is_dir():
                    directories.append(skill_dir)
            
            # 检查是否到达git根目录
            if (current / ".git").exists():
                break
            
            # 向上一级
            parent = current.parent
            if parent == current:  # 已到根目录
                break
            current = parent
        
        # 2. 全局级目录
        if self.enable_global:
            for global_path in self.GLOBAL_PATHS:
                if global_path.exists() and global_path.is_dir():
                    directories.append(global_path)
        
        logger.info(f"发现{len(directories)}个Skill目录")
        return directories
    
    def find_skill_files(self, skill_dir: Path) -> List[Path]:
        """
        在指定目录中查找所有SKILL.md文件
        
        Args:
            skill_dir: Skill目录路径
        
        Returns:
            SKILL.md文件路径列表
        """
        skill_files = []
        
        if not skill_dir.exists():
            return skill_files
        
        # 遍历子目录，查找SKILL.md
        for subdir in skill_dir.iterdir():
            if not subdir.is_dir():
                continue
            
            skill_md = subdir / "SKILL.md"
            if skill_md.exists() and skill_md.is_file():
                skill_files.append(skill_md)
        
        return skill_files
    
    def parse_skill_file(self, skill_path: Path) -> Optional[SkillMetadata]:
        """
        解析SKILL.md文件
        
        Args:
            skill_path: SKILL.md文件路径
        
        Returns:
            SkillMetadata实例，如果解析失败返回None
        """
        try:
            content = skill_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.error(f"读取文件失败 {skill_path}: {e}")
            return None
        
        # 提取YAML frontmatter
        match = self.FRONTMATTER_PATTERN.match(content)
        if not match:
            logger.warning(f"未找到YAML frontmatter: {skill_path}")
            return None
        
        frontmatter_text = match.group(1)
        markdown_content = content[match.end():]
        
        # 解析YAML
        try:
            yaml_data = yaml.safe_load(frontmatter_text)
        except yaml.YAMLError as e:
            logger.error(f"YAML解析失败 {skill_path}: {e}")
            return None
        
        if not isinstance(yaml_data, dict):
            logger.error(f"YAML数据格式错误 {skill_path}: 期望dict，得到{type(yaml_data)}")
            return None
        
        # 创建SkillMetadata
        try:
            metadata = SkillMetadata.from_yaml_dict(yaml_data, skill_path=skill_path)
            metadata.content = markdown_content.strip()
        except Exception as e:
            logger.error(f"创建SkillMetadata失败 {skill_path}: {e}")
            return None
        
        # 验证元数据
        is_valid, error_msg = metadata.validate()
        if not is_valid:
            logger.error(f"Skill元数据验证失败 {skill_path}: {error_msg}")
            return None
        
        return metadata
    
    def load_from_directory(self, skill_dir: Path) -> List[SkillMetadata]:
        """
        从指定目录加载所有Skill
        
        Args:
            skill_dir: Skill目录路径
        
        Returns:
            成功加载的SkillMetadata列表
        """
        skills = []
        
        # 查找SKILL.md文件
        skill_files = self.find_skill_files(skill_dir)
        logger.info(f"在 {skill_dir} 中找到{len(skill_files)}个SKILL.md文件")
        
        # 解析每个文件
        for skill_file in skill_files:
            # 避免重复加载
            if skill_file in self._loaded_paths:
                logger.debug(f"跳过已加载的Skill: {skill_file}")
                continue
            
            metadata = self.parse_skill_file(skill_file)
            if metadata:
                skills.append(metadata)
                self._loaded_paths.add(skill_file)
                logger.info(f"成功加载Skill: {metadata.name} ({skill_file})")
            else:
                logger.warning(f"加载Skill失败: {skill_file}")
        
        return skills
    
    def load_all(self, additional_dirs: Optional[List[Path]] = None) -> List[SkillMetadata]:
        """
        加载所有可用的Skill
        
        Args:
            additional_dirs: 额外的Skill目录列表
        
        Returns:
            所有成功加载的SkillMetadata列表
        """
        all_skills = []
        
        # 1. 发现标准目录
        directories = self.discover_skill_directories()
        
        # 2. 添加额外目录
        if additional_dirs:
            directories.extend(additional_dirs)
        
        # 3. 从每个目录加载
        for skill_dir in directories:
            try:
                skills = self.load_from_directory(skill_dir)
                all_skills.extend(skills)
            except Exception as e:
                logger.error(f"从 {skill_dir} 加载Skill时出错: {e}")
        
        logger.info(f"总共成功加载{len(all_skills)}个Skill")
        return all_skills
    
    def load_single(self, skill_path: Path) -> Optional[SkillMetadata]:
        """
        加载单个SKILL.md文件
        
        Args:
            skill_path: SKILL.md文件路径
        
        Returns:
            SkillMetadata实例，如果失败返回None
        """
        if not skill_path.exists():
            raise SkillLoadError(f"文件不存在: {skill_path}")
        
        if skill_path.name != "SKILL.md":
            raise SkillLoadError(f"不是SKILL.md文件: {skill_path}")
        
        metadata = self.parse_skill_file(skill_path)
        if metadata:
            self._loaded_paths.add(skill_path)
        
        return metadata
    
    def reload(self) -> List[SkillMetadata]:
        """
        重新加载所有Skill（清除缓存）
        
        Returns:
            重新加载的SkillMetadata列表
        """
        self._loaded_paths.clear()
        return self.load_all()
    
    def get_loaded_paths(self) -> Set[Path]:
        """获取已加载的Skill文件路径集合"""
        return self._loaded_paths.copy()

