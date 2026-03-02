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
Skill 筛选示例（无需 LLM）

演示如何使用 OperatorSkillSelector 进行纯规则筛选，无需 LLM：
1. 基于 metadata（backend, dsl, hardware, operator_type）筛选
2. 基于 category（include_categories, exclude_categories）筛选
3. 组合使用：metadata + category 双重筛选

所有筛选都在 coarse_filter 阶段完成，不依赖 LLM。

运行方式：
    cd /path/to/akg_agents
    conda activate akg_agents
    source env.sh
    python examples/run_skill/09_no_llm_filtering.py
"""

from pathlib import Path
from typing import List

from akg_agents.core_v2.skill import SkillRegistry, SkillMetadata
from akg_agents.op.skill import OperatorSkillSelector, OperatorSelectionContext


def print_skills(skills: List[SkillMetadata], title: str = "Skills"):
    """格式化打印 Skills 列表"""
    print(f"\n{title} ({len(skills)} 个):")
    if not skills:
        print("  (无)")
        return
    
    for skill in skills:
        cat_str = f"[{skill.category}]" if skill.category else "[--]"
        print(f"  {cat_str} {skill.name}")
        if skill.metadata:
            # 只显示关键 metadata
            meta_items = []
            for key in ['backend', 'dsl', 'operator_patterns']:
                if key in skill.metadata:
                    meta_items.append(f"{key}={skill.metadata[key]}")
            if meta_items:
                print(f"      {', '.join(meta_items)}")


def example_1_metadata_only():
    """示例1：仅使用 metadata 筛选"""
    print("=" * 70)
    print("示例1：仅使用 metadata 筛选（backend + dsl）")
    print("=" * 70)
    
    # 加载 Skills（从真实的 triton-ascend 目录）
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "resources" / "skills" / "triton-ascend"
    
    if not skills_dir.exists():
        print(f"[警告] 目录不存在: {skills_dir}")
        print("使用示例 skills 目录作为备选")
        skills_dir = Path(__file__).parent / "skills"
    
    count = registry.load_from_directory(skills_dir)
    print(f"加载了 {count} 个 Skills")
    
    # 创建选择器
    selector = OperatorSkillSelector()
    all_skills = registry.get_all()
    
    # 筛选条件：只要 ascend 后端 + triton-ascend DSL
    context = OperatorSelectionContext(
        backend="ascend",
        dsl="triton-ascend"
    )
    
    print(f"\n筛选条件:")
    print(f"  backend = {context.backend}")
    print(f"  dsl = {context.dsl}")
    
    # 执行筛选
    filtered = selector.coarse_filter(all_skills, context)
    
    print_skills(filtered, "筛选结果")
    print()


def example_2_category_only():
    """示例2：仅使用 category 筛选"""
    print("=" * 70)
    print("示例2：仅使用 category 筛选（include_categories）")
    print("=" * 70)
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "resources" / "skills" / "triton-ascend"
    
    if not skills_dir.exists():
        skills_dir = Path(__file__).parent / "skills"
    
    count = registry.load_from_directory(skills_dir)
    print(f"加载了 {count} 个 Skills")
    
    # 创建选择器
    selector = OperatorSkillSelector()
    all_skills = registry.get_all()
    
    # 筛选条件：只要 guide 和 example 分类
    context = OperatorSelectionContext(
        include_categories=["guide", "example"]
    )
    
    print(f"\n筛选条件:")
    print(f"  include_categories = [guide, example]")
    
    # 执行筛选
    filtered = selector.coarse_filter(all_skills, context)
    
    print_skills(filtered, "筛选结果")
    print()


def example_3_category_exclude():
    """示例3：使用 category 排除筛选"""
    print("=" * 70)
    print("示例3：使用 category 排除筛选（exclude_categories）")
    print("=" * 70)
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "resources" / "skills" / "triton-ascend"
    
    if not skills_dir.exists():
        skills_dir = Path(__file__).parent / "skills"
    
    count = registry.load_from_directory(skills_dir)
    print(f"加载了 {count} 个 Skills")
    
    # 创建选择器
    selector = OperatorSkillSelector()
    all_skills = registry.get_all()
    
    # 筛选条件：排除 workflow, agent, implementation（即只保留 guide, example 等）
    context = OperatorSelectionContext(
        exclude_categories=["workflow", "agent", "implementation"]
    )
    
    print(f"\n筛选条件:")
    print(f"  exclude_categories = [workflow, agent, implementation]")
    
    # 执行筛选
    filtered = selector.coarse_filter(all_skills, context)
    
    print_skills(filtered, "筛选结果")
    print()


def example_4_combined_filtering():
    """示例4：组合筛选（metadata + category 双重筛选）"""
    print("=" * 70)
    print("示例4：组合筛选（metadata + category 双重筛选）")
    print("=" * 70)
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "resources" / "skills" / "triton-ascend"
    
    if not skills_dir.exists():
        skills_dir = Path(__file__).parent / "skills"
    
    count = registry.load_from_directory(skills_dir)
    print(f"加载了 {count} 个 Skills")
    
    # 创建选择器
    selector = OperatorSkillSelector()
    all_skills = registry.get_all()
    
    # 组合筛选条件：
    # - metadata: backend=ascend, dsl=triton-ascend
    # - category: 只要 guide（基础文档）
    context = OperatorSelectionContext(
        backend="ascend",
        dsl="triton-ascend",
        include_categories=["guide"]
    )
    
    print(f"\n筛选条件:")
    print(f"  backend = {context.backend}")
    print(f"  dsl = {context.dsl}")
    print(f"  include_categories = [guide]")
    
    # 执行筛选
    filtered = selector.coarse_filter(all_skills, context)
    
    print_skills(filtered, "筛选结果（guide 基础文档）")
    
    # 再筛选：只要 example（具体案例）
    context2 = OperatorSelectionContext(
        backend="ascend",
        dsl="triton-ascend",
        include_categories=["example"]
    )
    
    print(f"\n筛选条件 2:")
    print(f"  backend = {context2.backend}")
    print(f"  dsl = {context2.dsl}")
    print(f"  include_categories = [example]")
    
    filtered2 = selector.coarse_filter(all_skills, context2)
    
    print_skills(filtered2, "筛选结果（example 具体案例）")
    print()


def example_5_include_and_exclude():
    """示例5：同时使用 include 和 exclude"""
    print("=" * 70)
    print("示例5：同时使用 include_categories 和 exclude_categories")
    print("=" * 70)
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "resources" / "skills" / "triton-ascend"
    
    if not skills_dir.exists():
        skills_dir = Path(__file__).parent / "skills"
    
    count = registry.load_from_directory(skills_dir)
    print(f"加载了 {count} 个 Skills")
    
    # 创建选择器
    selector = OperatorSkillSelector()
    all_skills = registry.get_all()
    
    # 复杂筛选条件：
    # - include_categories: 只在 guide, implementation, example 中选
    # - exclude_categories: 再排除 implementation
    # - 最终效果：只保留 guide, example
    context = OperatorSelectionContext(
        backend="ascend",
        dsl="triton-ascend",
        include_categories=["guide", "implementation", "example"],
        exclude_categories=["implementation"]
    )
    
    print(f"\n筛选条件:")
    print(f"  backend = {context.backend}")
    print(f"  dsl = {context.dsl}")
    print(f"  include_categories = [guide, implementation, example]")
    print(f"  exclude_categories = [implementation]")
    print(f"  → 最终效果：只保留 guide, example")
    
    # 执行筛选
    filtered = selector.coarse_filter(all_skills, context)
    
    print_skills(filtered, "筛选结果")
    print()


def example_6_build_prompt_without_llm():
    """示例6：无 LLM 构建完整 prompt"""
    print("=" * 70)
    print("示例6：无 LLM 构建完整 prompt")
    print("=" * 70)
    
    from akg_agents.core_v2.skill import build_prompt_with_skills
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "resources" / "skills" / "triton-ascend"
    
    if not skills_dir.exists():
        skills_dir = Path(__file__).parent / "skills"
    
    count = registry.load_from_directory(skills_dir)
    print(f"加载了 {count} 个 Skills")
    
    # 创建选择器
    selector = OperatorSkillSelector()
    all_skills = registry.get_all()
    
    # 筛选：只要 guide 基础文档
    context = OperatorSelectionContext(
        backend="ascend",
        dsl="triton-ascend",
        include_categories=["guide"]
    )
    
    print(f"\n筛选条件:")
    print(f"  backend = {context.backend}")
    print(f"  dsl = {context.dsl}")
    print(f"  include_categories = [guide]")
    
    # 执行筛选
    filtered = selector.coarse_filter(all_skills, context)
    
    print_skills(filtered, "选中的 Skills")
    
    # 构建 prompt（不使用 LLM）
    task_description = """
请使用 Triton Ascend 为以下 ReLU 算子生成高效实现：

输入:
- x: 输入张量，shape 为 (batch_size, dim)，dtype 为 float16

输出:
- y: 输出张量，y = max(0, x)

性能要求:
1. 使用 VEC 核心
2. BLOCK_SIZE = 1024
3. 支持大 batch_size
"""
    
    final_prompt = build_prompt_with_skills(
        filtered,
        task_description,
        include_full_content=False  # 只包含描述，避免 prompt 过长
    )
    
    print(f"\n生成的 prompt 长度: {len(final_prompt)} 字符")
    print("\nprompt 预览（前 800 字符）:")
    print("-" * 70)
    print(final_prompt[:800])
    print("...")
    print("-" * 70)
    print()


def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("=" * 70)
    print("Skill 筛选示例（无需 LLM）")
    print("=" * 70)
    print()
    
    examples = [
        ("示例1: 仅 metadata 筛选", example_1_metadata_only),
        ("示例2: 仅 category include 筛选", example_2_category_only),
        ("示例3: category exclude 筛选", example_3_category_exclude),
        ("示例4: 组合筛选", example_4_combined_filtering),
        ("示例5: include + exclude", example_5_include_and_exclude),
        ("示例6: 无 LLM 构建 prompt", example_6_build_prompt_without_llm),
    ]
    
    for name, example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n[错误] {name}: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    print("=" * 70)
    print("所有示例运行完成")
    print("=" * 70)


if __name__ == "__main__":
    run_all_examples()
