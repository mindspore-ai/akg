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
Skill System 层级管理示例

演示如何使用 Skill Hierarchy 管理 Skills 之间的层级关系：
1. 父子关系查询
2. 获取所有后代
3. 互斥组检查
4. 层级验证（可选功能）

运行方式：
    cd /path/to/akg_agents
    conda activate akg_agents
    source env.sh
    python examples/run_skill/03_skill_hierarchy.py
"""

from pathlib import Path

from akg_agents.core_v2.skill import (
    SkillRegistry, SkillHierarchy, SkillLevel,
    validate_all, detect_cycles
)


def example_1_basic_hierarchy():
    """示例1：基本的层级关系查询"""
    print("=" * 70)
    print("示例1：基本的层级关系查询")
    print("=" * 70)
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 创建 Hierarchy
    hierarchy = SkillHierarchy(registry)
    print(f"创建 Hierarchy: {hierarchy}\n")
    
    # 查询父子关系
    print("[1] 查询子 Skills:")
    test_skills = ["standard-workflow", "coder-agent", "adaptive-evolve"]
    
    for skill_name in test_skills:
        children = hierarchy.get_children(skill_name)
        if children:
            print(f"  {skill_name}:")
            for child in children:
                print(f"    - {child}")
        else:
            print(f"  {skill_name}: (无子 Skills)")
    
    print("\n[2] 查询父 Skills:")
    test_children = ["coder-agent", "verifier-agent", "cuda-basics"]
    
    for child_name in test_children:
        parents = hierarchy.get_parents(child_name)
        if parents:
            print(f"  {child_name}:")
            for parent in parents:
                print(f"    - {parent}")
        else:
            print(f"  {child_name}: (无父 Skills)")
    
    print()


def example_2_descendants():
    """示例2：获取所有后代"""
    print("=" * 70)
    print("示例2：获取所有后代")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    hierarchy = SkillHierarchy(registry)
    
    # 获取 standard-workflow 的所有后代
    print("\n[1] standard-workflow 的所有后代:")
    descendants = hierarchy.get_descendants("standard-workflow")
    for desc in descendants:
        skill = registry.get(desc)
        if skill:
            level_str = f"[{skill.level.value}]" if skill.level else "[--]"
            print(f"  {level_str} {desc}")
    
    # 获取 adaptive-evolve 的所有后代
    print("\n[2] adaptive-evolve 的所有后代:")
    descendants = hierarchy.get_descendants("adaptive-evolve")
    for desc in descendants:
        skill = registry.get(desc)
        if skill:
            level_str = f"[{skill.level.value}]" if skill.level else "[--]"
            print(f"  {level_str} {desc}")
    
    print()


def example_3_exclusive_groups():
    """示例3：互斥组检查"""
    print("=" * 70)
    print("示例3：互斥组检查")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    hierarchy = SkillHierarchy(registry)
    
    # 模拟场景：同时激活多个 Skills
    print("\n[测试场景] 同时激活多个 Skills:")
    
    # 场景1：标准组合（无冲突）
    active_skills_1 = {"standard-workflow", "coder-agent", "verifier-agent"}
    print(f"\n场景1: {active_skills_1}")
    
    for skill_name in active_skills_1:
        skill = registry.get(skill_name)
        if skill:
            conflict = hierarchy.check_exclusive_conflict(skill, active_skills_1)
            if conflict:
                print(f"  ✗ {skill_name}: {conflict}")
            else:
                print(f"  ✓ {skill_name}: 无冲突")
    
    # 场景2：测试其他组合
    test_skill = registry.get("coder-agent")
    if test_skill:
        active_skills_2 = {"designer-agent", "verifier-agent"}
        print(f"\n场景2: 激活 {active_skills_2}，尝试添加 coder-agent:")
        conflict = hierarchy.check_exclusive_conflict(test_skill, active_skills_2)
        if conflict:
            print(f"  ✗ {conflict}")
        else:
            print(f"  ✓ 无冲突")
    
    print()


def example_4_traverse_hierarchy():
    """示例4：遍历层级树"""
    print("=" * 70)
    print("示例4：遍历层级树")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    hierarchy = SkillHierarchy(registry)
    
    # 找到所有 L1 Skills（根节点）
    l1_skills = registry.get_by_level(SkillLevel.L1)
    
    print("\n层级树结构:")
    for l1_skill in l1_skills:
        print(f"\n[L1] {l1_skill.name}")
        print(f"     {l1_skill.description}")
        
        # 获取子 Skills
        children = hierarchy.get_children(l1_skill.name)
        for child_name in children:
            child = registry.get(child_name)
            if child:
                level_str = f"[{child.level.value}]" if child.level else "[--]"
                print(f"  ├── {level_str} {child.name}")
                
                # 获取孙子 Skills
                grandchildren = hierarchy.get_children(child_name)
                for i, gc_name in enumerate(grandchildren):
                    gc = registry.get(gc_name)
                    if gc:
                        gc_level_str = f"[{gc.level.value}]" if gc.level else "[--]"
                        is_last = i == len(grandchildren) - 1
                        prefix = "      └──" if is_last else "      ├──"
                        print(f"{prefix} {gc_level_str} {gc.name}")
    
    print()


def example_5_get_skill_path():
    """示例5：获取 Skill 路径（从根到叶子）"""
    print("=" * 70)
    print("示例5：获取 Skill 路径")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    hierarchy = SkillHierarchy(registry)
    
    # 定义辅助函数
    def find_path_to_root(skill_name):
        """找到从 skill 到根的路径"""
        path = [skill_name]
        current = skill_name
        
        while True:
            parents = hierarchy.get_parents(current)
            if not parents:
                break
            # 取第一个父节点（如果有多个父节点）
            parent = parents[0]
            path.append(parent)
            current = parent
        
        return list(reversed(path))
    
    # 测试几个 Skills 的路径
    test_skills = ["cuda-basics", "coder-agent", "standard-workflow"]
    
    print("\nSkills 到根的路径:")
    for skill_name in test_skills:
        path = find_path_to_root(skill_name)
        
        print(f"\n{skill_name}:")
        for i, node in enumerate(path):
            skill = registry.get(node)
            if skill:
                level_str = f"[{skill.level.value}]" if skill.level else "[--]"
                indent = "  " * i
                arrow = " → " if i < len(path) - 1 else ""
                print(f"{indent}{level_str} {node}{arrow}")
    
    print()


def example_6_validation():
    """示例6：层级验证（可选功能）"""
    print("=" * 70)
    print("示例6：层级验证（可选功能）")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    hierarchy = SkillHierarchy(registry)
    
    print("\n[1] 检测循环依赖:")
    cycles = detect_cycles(hierarchy)
    if cycles:
        print(f"  ✗ 发现 {len(cycles)} 个循环:")
        for cycle in cycles:
            print(f"    {' → '.join(cycle)}")
    else:
        print("  ✓ 无循环依赖")
    
    print("\n[2] 完整验证:")
    is_valid, errors = validate_all(hierarchy)
    if is_valid:
        print("  ✓ 所有层级关系验证通过")
    else:
        print(f"  ✗ 发现 {len(errors)} 个问题:")
        for error in errors:
            print(f"    - {error}")
    
    print()


def example_7_default_children():
    """示例7：默认子 Skills"""
    print("=" * 70)
    print("示例7：默认子 Skills")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 查找有默认子 Skills 的 Skills
    print("\n包含默认子 Skills 的 Skills:")
    
    all_skills = registry.get_all()
    for skill in all_skills:
        if skill.structure and skill.structure.default_children:
            print(f"\n[{skill.name}]")
            print(f"  所有子 Skills: {skill.structure.child_skills}")
            print(f"  默认子 Skills: {skill.structure.default_children}")
            
            # 显示差异
            non_default = set(skill.structure.child_skills) - set(skill.structure.default_children)
            if non_default:
                print(f"  可选子 Skills: {list(non_default)}")
    
    print()


def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("=" * 70)
    print("Skill System 层级管理示例")
    print("=" * 70)
    print()
    
    examples = [
        example_1_basic_hierarchy,
        example_2_descendants,
        example_3_exclusive_groups,
        example_4_traverse_hierarchy,
        example_5_get_skill_path,
        example_6_validation,
        example_7_default_children,
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
