# Copyright 2025 Huawei Technologies Co., Ltd
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
import re
import logging
from pathlib import Path
from typing import TypedDict, Optional, List, Dict

try:
    import yaml
except ImportError:
    yaml = None

logger = logging.getLogger(__name__)

DEFAULT_SKILLS_DIR = Path(__file__).parent.parent.parent / "resources" / "skills"


class SkillMetadata(TypedDict, total=False):
    """
    Skill 元数据，解析一下两个必要的字段
    """
    
    name: str
    """Skill 名称（小写字母、数字、连字符）"""
    
    description: str
    """Skill 描述"""
    
    path: str
    """SKILL.md 文件的完整路径"""


def parse_skill_metadata(skill_md_path: Path) -> Optional[SkillMetadata]:
    """
    解析 SKILL.md 文件的 YAML
    """
    if yaml is None:
        logger.warning("PyYAML not installed, cannot parse SKILL.md frontmatter")
        return None
    
    try:
        content = skill_md_path.read_text(encoding="utf-8")
        
        # 匹配 YAML frontmatter: --- ... ---，这里建议后者添加skill的时候遵循agentskill的规范！
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(frontmatter_pattern, content, re.DOTALL)
        
        if not match:
            logger.warning(f"No YAML frontmatter found in {skill_md_path}")
            return None
        
        frontmatter_data = yaml.safe_load(match.group(1))
        
        if not isinstance(frontmatter_data, dict):
            logger.warning(f"Invalid frontmatter format in {skill_md_path}")
            return None
        
        # 验证必需字段
        name = frontmatter_data.get("name")
        description = frontmatter_data.get("description")
        
        if not name or not description:
            logger.warning(f"Missing required fields (name/description) in {skill_md_path}")
            return None
        
        # 验证 name 格式
        if not re.match(r"^[a-z0-9][a-z0-9-]*[a-z0-9]$|^[a-z0-9]$", name):
            logger.warning(f"Invalid skill name format: {name}")
            return None
        
        return {
            "name": str(name),
            "description": str(description),
            "path": str(skill_md_path),
        }
        
    except Exception as e:
        logger.warning(f"Failed to parse {skill_md_path}: {e}")
        return None


def list_skills(skills_dir: Optional[Path] = None) -> List[SkillMetadata]:
    """
    扫描 skills 目录，加载所有 SKILL.md 文件的元数据
    
    """
    skills_dir = skills_dir or DEFAULT_SKILLS_DIR
    
    if not skills_dir.exists():
        logger.info(f"Skills directory not found: {skills_dir}")
        return []
    
    skills: List[SkillMetadata] = []
    
    for skill_subdir in skills_dir.iterdir():
        if not skill_subdir.is_dir():
            continue
        
        skill_md_path = skill_subdir / "SKILL.md"
        if not skill_md_path.exists():
            continue
        
        metadata = parse_skill_metadata(skill_md_path)
        if metadata:
            skills.append(metadata)
    
    logger.info(f"Loaded {len(skills)} skills from {skills_dir}")
    return skills


class SkillLoader:
    """
    Skills 加载器
    """
    
    def __init__(self, skills_dir: Optional[Path] = None):
        self.skills_dir = skills_dir or DEFAULT_SKILLS_DIR
        self._skills: List[SkillMetadata] = []
        self._skills_by_name: Dict[str, SkillMetadata] = {}
        
        self._load_skills()
    
    def _load_skills(self) -> None:
        self._skills = list_skills(self.skills_dir)
        self._skills_by_name = {s["name"]: s for s in self._skills}
        logger.info(f"SkillLoader initialized with {len(self._skills)} skills")
    
    @property
    def skills(self) -> List[SkillMetadata]:
        """获取所有 skill 元数据"""
        return self._skills
    
    def find_skill(self, name: str) -> Optional[SkillMetadata]:
        """
        按名称查找 skill
        
        """
        return self._skills_by_name.get(name)
    
    def get_skill_names(self) -> List[str]:
        """获取所有 skill 名称列表"""
        return list(self._skills_by_name.keys())
