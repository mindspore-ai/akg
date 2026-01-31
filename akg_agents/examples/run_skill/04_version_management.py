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
Skill System 版本管理示例

演示如何管理 Skills 的多个版本：
1. 注册多个版本
2. 版本选择策略
3. 版本比较
4. 多仓库场景

运行方式：
    cd /path/to/akg_agents
    conda activate akg_agents
    source env.sh
    python examples/run_skill/04_version_management.py
"""

from pathlib import Path

from akg_agents.core_v2.skill import (
    SkillMetadata, SkillLevel, SkillRegistry,
    Version, compare_versions
)


def example_1_version_basics():
    """示例1：版本基础操作"""
    print("=" * 70)
    print("示例1：版本基础操作")
    print("=" * 70)
    
    # 创建 Registry
    registry = SkillRegistry()
    
    # 创建同一 Skill 的多个版本
    v1 = SkillMetadata(
        name="cuda-basics",
        description="CUDA 基础知识 v1.0.0",
        level=SkillLevel.L3,
        version="1.0.0",
        content="# CUDA 基础 v1.0.0\n\n初始版本..."
    )
    
    v2 = SkillMetadata(
        name="cuda-basics",
        description="CUDA 基础知识 v2.0.0",
        level=SkillLevel.L3,
        version="2.0.0",
        content="# CUDA 基础 v2.0.0\n\n增加新特性..."
    )
    
    v3 = SkillMetadata(
        name="cuda-basics",
        description="CUDA 基础知识 v2.1.0",
        level=SkillLevel.L3,
        version="2.1.0",
        content="# CUDA 基础 v2.1.0\n\n修复bug..."
    )
    
    # 注册所有版本
    registry.register(v1)
    registry.register(v2)
    registry.register(v3)
    
    print("\n[1] 已注册的版本:")
    versions = registry.get_versions("cuda-basics")
    for ver in versions:
        print(f"  - {ver}")
    
    print(f"\n[2] 总共注册了 {len(versions)} 个版本")
    print()


def example_2_version_strategies():
    """示例2：版本选择策略"""
    print("=" * 70)
    print("示例2：版本选择策略")
    print("=" * 70)
    
    registry = SkillRegistry()
    
    # 注册多个版本
    versions = ["1.0.0", "1.5.0", "2.0.0", "2.1.0"]
    for ver in versions:
        skill = SkillMetadata(
            name="triton-syntax",
            description=f"Triton 语法 v{ver}",
            level=SkillLevel.L3,
            version=ver,
            content=f"# Triton v{ver}"
        )
        registry.register(skill)
    
    print("\n[1] 获取最新版本（默认）:")
    latest = registry.get("triton-syntax")  # strategy="latest" 是默认值
    print(f"  版本: {latest.version}")
    print(f"  描述: {latest.description}")
    
    print("\n[2] 获取最旧/稳定版本:")
    oldest = registry.get("triton-syntax", strategy="oldest")
    print(f"  版本: {oldest.version}")
    print(f"  描述: {oldest.description}")
    
    print("\n[3] 获取指定版本:")
    specific = registry.get("triton-syntax", version="2.0.0")
    if specific:
        print(f"  版本: {specific.version}")
        print(f"  描述: {specific.description}")
    
    print("\n[4] 使用场景:")
    print("  - 生产环境: strategy='oldest' (稳定版本)")
    print("  - 测试环境: strategy='latest' (最新功能)")
    print("  - 特定需求: version='x.y.z' (精确版本)")
    print()


def example_3_version_comparison():
    """示例3：版本比较"""
    print("=" * 70)
    print("示例3：版本比较")
    print("=" * 70)
    
    # 测试版本比较
    test_cases = [
        ("1.0.0", "2.0.0"),
        ("2.0.0", "2.1.0"),
        ("2.1.0", "2.1.0"),
        ("1.9.0", "2.0.0"),
        ("2.0.0-alpha", "2.0.0"),
    ]
    
    print("\n版本比较结果:")
    for v1, v2 in test_cases:
        result = compare_versions(v1, v2)
        if result < 0:
            symbol = "<"
        elif result > 0:
            symbol = ">"
        else:
            symbol = "=="
        print(f"  {v1:15} {symbol} {v2}")
    
    print("\n[说明] 使用语义化版本（SemVer）规则")
    print("  - MAJOR.MINOR.PATCH")
    print("  - 预发布版本（alpha/beta）< 正式版本")
    print()


def example_4_multi_repo_scenario():
    """示例4：多仓库场景"""
    print("=" * 70)
    print("示例4：多仓库场景")
    print("=" * 70)
    
    registry = SkillRegistry()
    
    # 模拟场景：从两个仓库加载 Skills
    print("\n[场景] 维护两个 Skills 仓库:")
    print("  - stable_repo: 稳定版本（v1.x.x）")
    print("  - latest_repo: 最新版本（v2.x.x）")
    
    # 稳定仓库的 Skills
    stable_skills = [
        ("cuda-basics", "1.0.0"),
        ("triton-syntax", "1.5.0"),
    ]
    
    print("\n[1] 加载稳定仓库:")
    for name, ver in stable_skills:
        skill = SkillMetadata(
            name=name,
            description=f"{name} v{ver} (stable)",
            level=SkillLevel.L3,
            version=ver
        )
        registry.register(skill)
        print(f"  ✓ {name} v{ver}")
    
    # 最新仓库的 Skills（部分升级）
    latest_skills = [
        ("cuda-basics", "2.0.0"),  # 升级
        # triton-syntax 保持 1.5.0
    ]
    
    print("\n[2] 加载最新仓库（部分升级）:")
    for name, ver in latest_skills:
        skill = SkillMetadata(
            name=name,
            description=f"{name} v{ver} (latest)",
            level=SkillLevel.L3,
            version=ver
        )
        registry.register(skill)
        print(f"  ✓ {name} v{ver}")
    
    # 查询结果
    print("\n[3] 版本选择:")
    for name in ["cuda-basics", "triton-syntax"]:
        versions = registry.get_versions(name)
        latest = registry.get(name, strategy="latest")
        oldest = registry.get(name, strategy="oldest")
        
        print(f"\n  {name}:")
        print(f"    所有版本: {versions}")
        print(f"    最新版本: {latest.version}")
        print(f"    稳定版本: {oldest.version}")
    
    print()


def example_5_version_statistics():
    """示例5：版本统计信息"""
    print("=" * 70)
    print("示例5：版本统计信息")
    print("=" * 70)
    
    registry = SkillRegistry()
    
    # 加载真实的 Skills
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 手动添加一些多版本的 Skills
    for ver in ["1.0.0", "2.0.0"]:
        skill = SkillMetadata(
            name="test-skill",
            description=f"Test v{ver}",
            level=SkillLevel.L3,
            version=ver
        )
        registry.register(skill)
    
    # 获取统计信息
    stats = registry.get_statistics()
    
    print("\nRegistry 统计:")
    print(f"  总 Skills 数: {stats['total']}")
    print(f"  总版本数: {stats['total_versions']}")
    print(f"  多版本 Skills: {stats['multi_version_skills']}")
    
    # 查找多版本的 Skills
    print("\n多版本 Skills 详情:")
    for name in registry.get_names():
        versions = registry.get_versions(name)
        if len(versions) > 1:
            print(f"  {name}: {versions}")
    
    print()


def example_6_version_queries():
    """示例6：高级版本查询"""
    print("=" * 70)
    print("示例6：高级版本查询")
    print("=" * 70)
    
    registry = SkillRegistry()
    
    # 注册多个版本
    test_versions = [
        "1.0.0", "1.1.0", "1.2.0",
        "2.0.0", "2.1.0",
        "3.0.0-alpha", "3.0.0"
    ]
    
    for ver in test_versions:
        skill = SkillMetadata(
            name="advanced-skill",
            description=f"Advanced v{ver}",
            level=SkillLevel.L3,
            version=ver
        )
        registry.register(skill)
    
    print("\n[1] 所有版本（按时间排序）:")
    all_versions = registry.get_all_versions("advanced-skill")
    for skill in all_versions:
        print(f"  - v{skill.version}: {skill.description}")
    
    print("\n[2] 主要版本:")
    major_versions = {}
    for skill in all_versions:
        try:
            ver = Version.parse(skill.version)
            major = ver.major
            if major not in major_versions or ver > Version.parse(major_versions[major].version):
                major_versions[major] = skill
        except:
            pass
    
    for major, skill in sorted(major_versions.items()):
        print(f"  v{major}.x.x: {skill.version}")
    
    print()


def example_7_practical_workflow():
    """示例7：实际工作流"""
    print("=" * 70)
    print("示例7：实际工作流")
    print("=" * 70)
    
    print("\n[实际使用场景]")
    print("""
1. 开发阶段:
   - 从 skills_dev/ 加载最新版本
   - 使用 strategy='latest'
   
2. 测试阶段:
   - 从 skills_test/ 加载
   - 混合使用 latest 和 oldest
   
3. 生产阶段:
   - 从 skills_prod/ 加载稳定版本
   - 使用 strategy='oldest'
   - 或指定 version='x.y.z'
    """)
    
    # 示例代码
    registry = SkillRegistry()
    
    print("\n[代码示例]")
    print("```python")
    print("# 生产环境：使用稳定版本")
    print("cuda = registry.get('cuda-basics', strategy='oldest')")
    print()
    print("# 测试环境：使用最新版本")
    print("triton = registry.get('triton-syntax', strategy='latest')")
    print()
    print("# 特定需求：指定版本")
    print("specific = registry.get('coder-agent', version='2.0.0')")
    print("```")
    print()


def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("=" * 70)
    print("Skill System 版本管理示例")
    print("=" * 70)
    print()
    
    examples = [
        example_1_version_basics,
        example_2_version_strategies,
        example_3_version_comparison,
        example_4_multi_repo_scenario,
        example_5_version_statistics,
        example_6_version_queries,
        example_7_practical_workflow,
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"[错误] {example.__name__}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 70)
    print("所有示例运行完成")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
