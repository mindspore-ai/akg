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
Skill System 单元测试

测试 SkillMetadata、SkillRegistry、Version 等核心功能
"""

import pytest
from pathlib import Path

from ai_kernel_generator.core_v2.skill import (
    SkillMetadata,
    SkillLevel,
    SkillStructure,
    SkillRegistry,
    Version,
    VersionManager,
    compare_versions,
)


class TestSkillMetadata:
    """测试 SkillMetadata 元数据"""
    
    def test_create_minimal_skill(self):
        """创建最小 Skill（仅必需字段）"""
        skill = SkillMetadata(
            name="test-skill",
            description="测试 Skill"
        )
        
        assert skill.name == "test-skill"
        assert skill.description == "测试 Skill"
        assert skill.version == "1.0.0"
        assert skill.level is None
    
    def test_create_full_skill(self):
        """创建完整 Skill（所有字段）"""
        skill = SkillMetadata(
            name="full-skill",
            description="完整的测试 Skill",
            level=SkillLevel.L3,
            version="2.1.0",
            category="dsl",
            license="MIT",
            metadata={"backend": "cuda"},
            skill_path=Path("./test.md"),
            content="# Test Content"
        )
        
        assert skill.name == "full-skill"
        assert skill.level == SkillLevel.L3
        assert skill.version == "2.1.0"
        assert skill.category == "dsl"
        assert skill.metadata["backend"] == "cuda"
    
    def test_validate_valid_skill(self):
        """验证有效 Skill"""
        skill = SkillMetadata(
            name="valid-skill",
            description="这是一个有效的描述"
        )
        
        is_valid, error = skill.validate()
        assert is_valid is True
        assert error is None
    
    def test_validate_invalid_name(self):
        """验证无效名称格式"""
        # 包含大写字母
        skill1 = SkillMetadata(name="InvalidName", description="测试")
        is_valid1, _ = skill1.validate()
        assert is_valid1 is False
        
        # 包含空格
        skill2 = SkillMetadata(name="invalid name", description="测试")
        is_valid2, _ = skill2.validate()
        assert is_valid2 is False
        
        # 包含下划线
        skill3 = SkillMetadata(name="invalid_name", description="测试")
        is_valid3, _ = skill3.validate()
        assert is_valid3 is False
    
    def test_validate_valid_names(self):
        """验证有效名称格式"""
        valid_names = ["skill", "my-skill", "skill-123", "a-b-c"]
        
        for name in valid_names:
            skill = SkillMetadata(name=name, description="测试")
            is_valid, _ = skill.validate()
            assert is_valid is True, f"'{name}' 应该是有效的"
    
    def test_from_yaml_dict(self):
        """从 YAML 字典创建"""
        yaml_data = {
            "name": "yaml-skill",
            "description": "从 YAML 创建",
            "level": "L2",
            "version": "1.5.0",
            "category": "agent"
        }
        
        skill = SkillMetadata.from_yaml_dict(yaml_data)
        
        assert skill.name == "yaml-skill"
        assert skill.level == SkillLevel.L2
        assert skill.version == "1.5.0"
        assert skill.category == "agent"
    
    def test_from_yaml_dict_with_structure(self):
        """从 YAML 字典创建（含层级结构）"""
        yaml_data = {
            "name": "workflow-skill",
            "description": "工作流",
            "level": "L1",
            "structure": {
                "child_skills": ["skill1", "skill2"],
                "default_children": ["skill1"]
            }
        }
        
        skill = SkillMetadata.from_yaml_dict(yaml_data)
        
        assert skill.structure is not None
        assert skill.structure.child_skills == ["skill1", "skill2"]
        assert skill.structure.default_children == ["skill1"]


class TestSkillRegistry:
    """测试 SkillRegistry 注册表"""
    
    def test_register_and_get(self):
        """注册和查询 Skill"""
        registry = SkillRegistry()
        
        skill = SkillMetadata(
            name="test-skill",
            description="测试",
            level=SkillLevel.L3
        )
        registry.register(skill)
        
        retrieved = registry.get("test-skill")
        assert retrieved is not None
        assert retrieved.name == "test-skill"
    
    def test_get_by_level(self):
        """按层级查询"""
        registry = SkillRegistry()
        
        registry.register(SkillMetadata(name="l1-skill", description="L1", level=SkillLevel.L1))
        registry.register(SkillMetadata(name="l2-skill", description="L2", level=SkillLevel.L2))
        registry.register(SkillMetadata(name="l3-skill", description="L3", level=SkillLevel.L3))
        
        l2_skills = registry.get_by_level(SkillLevel.L2)
        assert len(l2_skills) == 1
        assert l2_skills[0].name == "l2-skill"
    
    def test_filter_by_name_pattern(self):
        """按名称模式过滤"""
        registry = SkillRegistry()
        
        registry.register(SkillMetadata(
            name="cuda-skill", 
            description="CUDA",
            level=SkillLevel.L3
        ))
        registry.register(SkillMetadata(
            name="cuda-basics",
            description="CUDA Basics",
            level=SkillLevel.L3
        ))
        registry.register(SkillMetadata(
            name="ascend-skill",
            description="Ascend",
            level=SkillLevel.L3
        ))
        
        cuda_skills = registry.filter(name_pattern="cuda-*")
        assert len(cuda_skills) == 2
        assert all(s.name.startswith("cuda-") for s in cuda_skills)
    
    def test_multi_version_support(self):
        """多版本支持"""
        registry = SkillRegistry()
        
        registry.register(SkillMetadata(name="test", description="v1", version="1.0.0", level=SkillLevel.L3))
        registry.register(SkillMetadata(name="test", description="v2", version="2.0.0", level=SkillLevel.L3))
        
        # 默认获取最新版本
        latest = registry.get("test")
        assert latest.version == "2.0.0"
        
        # 获取指定版本
        v1 = registry.get("test", version="1.0.0")
        assert v1.version == "1.0.0"
        
        # 获取所有版本
        all_versions = registry.get_all_versions("test")
        assert len(all_versions) == 2
    
    def test_version_strategies(self):
        """版本选择策略"""
        registry = SkillRegistry()
        
        registry.register(SkillMetadata(name="test", description="v1", version="1.0.0", level=SkillLevel.L3))
        registry.register(SkillMetadata(name="test", description="v2", version="2.0.0", level=SkillLevel.L3))
        
        # 策略：latest
        latest = registry.get("test", strategy="latest")
        assert latest.version == "2.0.0"
        
        # 策略：oldest
        oldest = registry.get("test", strategy="oldest")
        assert oldest.version == "1.0.0"


class TestVersion:
    """测试 Version 版本管理"""
    
    def test_parse_simple_version(self):
        """解析简单版本"""
        v = Version.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
    
    def test_parse_prerelease(self):
        """解析预发布版本"""
        v = Version.parse("1.0.0-alpha.1")
        assert v.major == 1
        assert v.prerelease == "alpha.1"
    
    def test_version_comparison(self):
        """版本比较"""
        v1 = Version.parse("1.0.0")
        v2 = Version.parse("1.0.1")
        v3 = Version.parse("2.0.0")
        
        assert v1 < v2
        assert v2 < v3
        assert v3 > v1
    
    def test_prerelease_comparison(self):
        """预发布版本比较"""
        v1 = Version.parse("1.0.0-alpha")
        v2 = Version.parse("1.0.0-beta")
        v3 = Version.parse("1.0.0")
        
        assert v1 < v2
        assert v2 < v3  # 预发布 < 正式版本


class TestVersionManager:
    """测试 VersionManager"""
    
    @pytest.fixture
    def manager(self):
        """创建版本管理器"""
        return VersionManager()
    
    def test_register_and_get(self, manager):
        """注册和获取"""
        skill = SkillMetadata(name="test", description="Test", version="1.0.0", level=SkillLevel.L3)
        manager.register_skill(skill)
        
        retrieved = manager.get_skill("test")
        assert retrieved is not None
        assert retrieved.version == "1.0.0"
    
    def test_multiple_versions(self, manager):
        """多版本管理"""
        manager.register_skill(SkillMetadata(name="cuda", description="v1", version="1.0.0", level=SkillLevel.L3))
        manager.register_skill(SkillMetadata(name="cuda", description="v2", version="1.5.0", level=SkillLevel.L3))
        manager.register_skill(SkillMetadata(name="cuda", description="v3", version="2.0.0", level=SkillLevel.L3))
        
        versions = manager.get_versions("cuda")
        assert len(versions) == 3
        assert versions == ["1.0.0", "1.5.0", "2.0.0"]
        
        latest = manager.get_latest_version("cuda")
        assert latest == "2.0.0"
        
        oldest = manager.get_oldest_version("cuda")
        assert oldest == "1.0.0"
    
    def test_version_strategies(self, manager):
        """版本选择策略"""
        manager.register_skill(SkillMetadata(name="test", description="v1", version="1.0.0", level=SkillLevel.L3))
        manager.register_skill(SkillMetadata(name="test", description="v2", version="2.0.0", level=SkillLevel.L3))
        
        latest = manager.get_skill("test", strategy="latest")
        assert latest.version == "2.0.0"
        
        oldest = manager.get_skill("test", strategy="oldest")
        assert oldest.version == "1.0.0"


class TestCompareVersions:
    """测试版本比较函数"""
    
    def test_compare_simple(self):
        """简单版本比较"""
        assert compare_versions("1.0.0", "2.0.0") < 0
        assert compare_versions("2.0.0", "1.0.0") > 0
        assert compare_versions("1.0.0", "1.0.0") == 0
    
    def test_compare_with_prerelease(self):
        """预发布版本比较"""
        assert compare_versions("1.0.0-alpha", "1.0.0") < 0
        assert compare_versions("1.0.0-alpha", "1.0.0-beta") < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
