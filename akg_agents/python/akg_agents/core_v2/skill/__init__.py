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
AIKG Skill Management System

一个符合 Agent Skills 开放标准的技能管理系统，支持：
- 标准 SKILL.md 格式（YAML Frontmatter + Markdown）
- 多路径加载（项目级 + 全局级）
- 统一安装位置（~/.akg/skills）
- 分级管理（L1-L5）+ 层级验证
- LLM 驱动的 Skill 智能选择
- 多版本管理（SemVer）
- URL安装（GitHub等）

设计理念：
- 层级用于组织（声明父子关系，必要的依赖检查）
- LLM 驱动决策（而非固定编排规则）
- 两阶段筛选（粗筛 + 精筛）
- 场景解耦（不绑定特定应用场景）
- 版本灵活性（支持多仓库、多版本选择策略）
- 统一路径（所有skills安装到标准位置）

参考标准：
- Agent Skills 开放标准 (agent-skills.cc)
- Claude Code Skills 规范
- AIKG UNIFIED_SKILL_FILESYSTEM_DESIGN
- Semantic Versioning 2.0.0

快速开始：
    >>> from skill_system import (
    ...     SkillRegistry, SkillInstaller, SkillHierarchy,
    ...     SkillSelector, SelectionContext,
    ...     VersionManager
    ... )
    >>> from pathlib import Path
    >>> 
    >>> # 1. 安装skills到统一位置
    >>> installer = SkillInstaller()
    >>> installer.install_from_directory(Path("./skills"))
    >>> 
    >>> # 2. 从安装位置加载
    >>> registry = SkillRegistry()
    >>> count = registry.load_from_directory(installer.install_root)
    >>> 
    >>> # 3. 版本选择策略
    >>> # 获取最新版本（测试环境）
    >>> cuda_latest = registry.get("cuda-basics", strategy="latest")
    >>> # 获取稳定版本（生产环境）
    >>> cuda_stable = registry.get("cuda-basics", strategy="oldest")
    >>> # 获取指定版本
    >>> cuda_v1 = registry.get("cuda-basics", version="1.0.0")
    >>> 
    >>> # 4. LLM 驱动选择
    >>> selector = SkillSelector()
    >>> context = SelectionContext(task_type="operator_generation")
    >>> selected = await selector.select(all_skills, context, llm_func)
"""

# 核心模块
from .metadata import SkillMetadata, SkillLevel, SkillStructure
from .loader import SkillLoader, SkillLoadError
from .registry import SkillRegistry
from .hierarchy import (
    SkillHierarchy,
    # 验证功能（可选）
    validate_all,
    detect_cycles
)
from .installer import SkillInstaller

# LLM 驱动选择（通用）
from .skill_selector import (
    SkillSelector,
    SelectionContext,
    build_prompt_with_skills,
    # 过滤器工具
    create_metadata_matcher,
    and_filters,
    or_filters
)

# 版本管理（简化版）
from .version import (
    Version,
    VersionManager,
    VersionStrategy,
    compare_versions
)
__all__ = [
    # 核心元数据
    "SkillMetadata",
    "SkillLevel",
    "SkillStructure",
    # 加载器
    "SkillLoader",
    "SkillLoadError",
    # 注册表
    "SkillRegistry",
    # 安装管理（支持URL安装）
    "SkillInstaller",
    # 层级管理
    "SkillHierarchy",
    # 层级验证（可选功能）
    "validate_all",
    "detect_cycles",
    # Skill 选择（通用）
    "SkillSelector",
    "SelectionContext",
    "build_prompt_with_skills",
    # 过滤器工具
    "create_metadata_matcher",
    "and_filters",
    "or_filters",
    # 版本管理
    "Version",
    "VersionManager",
    "VersionStrategy",
    "compare_versions",
]

