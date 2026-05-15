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
Skill System 基础使用示例

演示 Skill System 的基本功能：
1. 加载 Skills
2. 使用 Registry 管理
3. 查询和过滤
4. 查看 Skill 内容

运行方式：
    cd /path/to/akg_agents
    conda activate akg_agents
    source env.sh
    python examples/run_skill/01_basic_usage.py
"""

from pathlib import Path

from akg_agents.core_v2.skill import (
    SkillMetadata, SkillRegistry
)


def example_1_load_and_register():
    """示例1：加载并注册 Skills"""
    print("=" * 70)
    print("示例1：加载并注册 Skills")
    print("=" * 70)
    
    # 创建 Registry
    registry = SkillRegistry()
    
    # 从目录加载
    skills_dir = Path(__file__).parent / "skills"
    count = registry.load_from_directory(skills_dir)
    
    print(f"成功加载 {count} 个 Skills\n")
    
    # 查看统计信息
    stats = registry.get_statistics()
    print("Registry 统计:")
    print(f"  总数: {stats['total']}")
    print(f"  按分类:")
    for cat, cnt in stats['by_category'].items():
        print(f"    {cat}: {cnt} 个")
    print(f"  包含层级结构: {stats['with_structure']} 个")
    print()


def example_2_query_skills():
    """示例2：查询 Skills"""
    print("=" * 70)
    print("示例2：查询 Skills")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 按名称查询
    print("[1] 按名称查询:")
    skill = registry.get("cuda-basics")
    if skill:
        print(f"  名称: {skill.name}")
        print(f"  描述: {skill.description}")
        print(f"  分类: {skill.category or 'N/A'}")
        print(f"  版本: {skill.version}")
        print(f"  内容长度: {len(skill.content)} 字符")
    print()
    
    # 获取所有 Skills
    print("[2] 获取所有 Skills:")
    all_skills = registry.get_all()
    print(f"  总共: {len(all_skills)} 个")
    for skill in sorted(all_skills, key=lambda s: s.name)[:5]:
        cat_str = f"[{skill.category}]" if skill.category else "[--]"
        print(f"    {cat_str} {skill.name}")
    print()
    
    # 检查是否存在
    print("[3] 检查 Skill 是否存在:")
    print(f"  cuda-basics 存在: {registry.exists('cuda-basics')}")
    print(f"  non-existent 存在: {registry.exists('non-existent')}")
    print()


def example_3_filter_by_category():
    """示例3：按分类过滤"""
    print("=" * 70)
    print("示例3：按分类过滤")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 查询不同分类
    for category in ["workflow", "agent", "guide", "dsl"]:
        skills = registry.get_by_category(category)
        if skills:
            print(f"\n[{category}] {len(skills)} 个 Skills:")
            for skill in skills:
                print(f"  - {skill.name}: {skill.description[:50]}...")


def example_4_filter_by_pattern():
    """示例4：按名称模式过滤"""
    print("\n" + "=" * 70)
    print("示例4：按名称模式过滤")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 查找所有 agent skills
    print("\n[1] 名称包含 'agent' 的 Skills:")
    agent_skills = registry.filter(name_pattern="*agent*")
    for skill in agent_skills:
        cat_str = f"[{skill.category}]" if skill.category else "[--]"
        print(f"  {cat_str} {skill.name}")
    
    # 查找所有 triton 相关
    print("\n[2] 名称包含 'triton' 的 Skills:")
    triton_skills = registry.filter(name_pattern="*triton*")
    for skill in triton_skills:
        cat_str = f"[{skill.category}]" if skill.category else "[--]"
        print(f"  {cat_str} {skill.name}")
    
    # 查找有层级结构的 Skills
    print("\n[3] 包含层级结构的 Skills:")
    struct_skills = registry.filter(has_structure=True)
    for skill in struct_skills:
        print(f"  - {skill.name}")
        if skill.structure and skill.structure.child_skills:
            print(f"    子 Skills: {', '.join(skill.structure.child_skills)}")
    print()


def example_5_view_content():
    """示例5：查看 Skill 内容"""
    print("=" * 70)
    print("示例5：查看 Skill 内容")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 获取一个 Skill
    skill = registry.get("cuda-basics")
    
    if skill:
        print(f"\nSkill: {skill.name}")
        print(f"描述: {skill.description}")
        print(f"分类: {skill.category or 'N/A'}")
        print(f"版本: {skill.version}")
        print(f"文件路径: {skill.skill_path}")
        
        # 显示内容预览
        print(f"\n内容预览（前 500 字符）:")
        print("-" * 70)
        print(skill.content[:500])
        if len(skill.content) > 500:
            print("...")
        print("-" * 70)
        
        print(f"\n完整内容长度: {len(skill.content)} 字符")
    
    print()


def example_6_combined_filters():
    """示例6：组合过滤条件"""
    print("=" * 70)
    print("示例6：组合过滤条件")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # agent 分类 + 名称包含 'agent'
    print("\n[1] agent 分类且名称包含 'agent':")
    agent_skills = registry.filter(category="agent", name_pattern="*agent*")
    for skill in agent_skills:
        print(f"  - {skill.name}: {skill.description[:40]}...")
    
    # guide/dsl 分类 + 有层级结构
    print("\n[2] guide 分类且包含层级结构:")
    guide_structured = registry.filter(category="guide", has_structure=True)
    for skill in guide_structured:
        print(f"  - {skill.name}")
        if skill.structure:
            print(f"    子 Skills: {skill.structure.child_skills}")
    
    print()


def example_7_metadata_inspection():
    """示例7：检查 Skill 元数据"""
    print("=" * 70)
    print("示例7：检查 Skill 元数据")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 获取多个 Skills
    skills_to_inspect = ["standard-workflow", "coder-agent", "cuda-basics"]
    
    for skill_name in skills_to_inspect:
        skill = registry.get(skill_name)
        if skill:
            print(f"\n[{skill.name}]")
            print(f"  描述: {skill.description}")
            print(f"  分类: {skill.category or 'N/A'}")
            print(f"  版本: {skill.version}")
            print(f"  许可证: {skill.license if skill.license else 'N/A'}")
            
            # 自定义元数据
            if skill.metadata:
                print(f"  自定义元数据: {skill.metadata}")
            
            # 层级结构
            if skill.structure and skill.structure.child_skills:
                print(f"  子 Skills: {', '.join(skill.structure.child_skills)}")
    
    print()


def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("=" * 70)
    print("Skill System 基础使用示例")
    print("=" * 70)
    print()
    
    examples = [
        example_1_load_and_register,
        example_2_query_skills,
        example_3_filter_by_category,
        example_4_filter_by_pattern,
        example_5_view_content,
        example_6_combined_filters,
        example_7_metadata_inspection,
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
