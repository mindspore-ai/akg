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
Skill System 算子生成场景示例

演示如何使用 Skill System 完成算子生成任务：
1. 使用 OperatorSkillSelector 进行领域专用选择
2. L1 Workflow -> L2 Agent -> L3 DSL 的层级选择
3. LLM 驱动的智能选择
4. 拼接 prompt 生成代码

这是 Skill System 的核心应用场景，展示了完整的算子生成流程。

运行方式：
    cd /path/to/akg_agents
    conda activate akg_agents
    source env.sh
    python examples/run_skill/02_operator_generation.py

注意：本示例需要配置 LLM，参考 tools/v2/use_llm_check/test_run_llm.py
"""

from pathlib import Path
import asyncio

from akg_agents.core_v2.skill import (
    SkillRegistry, SkillLevel,
    build_prompt_with_skills
)
from akg_agents.op.skill import (
    OperatorSkillSelector,
    OperatorSelectionContext
)

# 导入 LLM 相关
try:
    from akg_agents.core_v2.agents import AgentBase, Jinja2TemplateWrapper, register_agent
    from akg_agents.core_v2.config import print_settings_info
    HAS_LLM = True
except ImportError:
    print("[警告] 无法导入 LLM 模块，将跳过需要 LLM 的示例")
    HAS_LLM = False


if HAS_LLM:
    @register_agent
    class OperatorGenerationAgent(AgentBase):
        """算子生成 Agent"""
        
        def __init__(self):
            context = {"agent_name": "operator_generation"}
            super().__init__(context=context)
        
        async def generate(self, prompt: str, model_level: str = "standard"):
            """生成文本"""
            template = Jinja2TemplateWrapper("{{ prompt }}")
            content, formatted_prompt, reasoning_content = await self.run_llm(
                template, {"prompt": prompt}, model_level
            )
            return content


def example_1_basic_operator_selection():
    """示例1：基础算子选择（不使用 LLM）"""
    print("=" * 70)
    print("示例1：基础算子选择（不使用 LLM）")
    print("=" * 70)
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 创建算子选择器
    selector = OperatorSkillSelector()
    
    # 定义算子生成上下文
    context = OperatorSelectionContext(
        operator_type="softmax",
        backend="cuda",
        dsl="triton"
    )
    
    print("\n任务上下文:")
    for k, v in context.to_dict().items():
        print(f"  {k}: {v}")
    
    # 进行粗筛（基于 metadata）
    all_skills = registry.get_all()
    candidates = selector.coarse_filter(all_skills, context)
    
    print(f"\n粗筛结果: {len(candidates)}/{len(all_skills)} 个 Skills")
    for skill in candidates:
        level_str = f"[{skill.level.value}]" if skill.level else "[--]"
        print(f"  {level_str} {skill.name}")
        if skill.metadata:
            print(f"      标签: {skill.metadata}")
    
    print()


async def example_2_llm_skill_selection():
    """示例2：使用 LLM 进行精筛"""
    print("=" * 70)
    print("示例2：使用 LLM 进行精筛")
    print("=" * 70)
    
    if not HAS_LLM:
        print("[跳过] 需要 LLM 支持\n")
        return
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 创建选择器和 Agent
    selector = OperatorSkillSelector()
    agent = OperatorGenerationAgent()
    
    # 定义上下文
    context = OperatorSelectionContext(
        operator_type="matmul",
        backend="cuda",
        dsl="triton"
    )
    
    print("\n任务上下文:")
    for k, v in context.to_dict().items():
        print(f"  {k}: {v}")
    
    # 粗筛
    all_skills = registry.get_all()
    candidates = selector.coarse_filter(all_skills, context)
    print(f"\n粗筛: {len(all_skills)} -> {len(candidates)} Skills")
    
    # LLM 精筛（用户自定义 prompt 模板）
    prompt_template = """你是一个 Skill 选择专家。

任务上下文：
{context_str}

候选 Skills（已粗筛）：
{skills_str}

请分析任务需求和候选 Skill，选择相关的Skill，保证不要有遗漏。

返回 JSON 格式：
```json
{{
  "selected": ["skill-name-1", "skill-name-2"],
  "reason": "选择理由"
}}
```
"""
    prompt = selector.build_llm_prompt(candidates, context, prompt_template)
    llm_response = await agent.generate(prompt)
    selected = selector.parse_llm_response(llm_response, candidates)
    
    if not selected:
        print("LLM 未选中任何 Skill，使用粗筛结果")
        selected = candidates
    
    print(f"\n精筛: {len(candidates)} -> {len(selected)} Skills")
    for skill in selected:
        level_str = f"[{skill.level.value}]" if skill.level else "[--]"
        print(f"  {level_str} {skill.name}: {skill.description[:50]}...")
    
    print()


async def example_3_complete_workflow():
    """示例3：完整的算子生成流程"""
    print("=" * 70)
    print("示例3：完整的算子生成流程")
    print("=" * 70)
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    all_skills = registry.get_all()
    
    print(f"\n加载了 {len(all_skills)} 个 Skills")
    
    # 定义任务
    operator_type = "softmax"
    backend = "cuda"
    dsl = "triton"
    
    print(f"\n任务: 生成 {operator_type} 算子")
    print(f"后端: {backend}")
    print(f"DSL: {dsl}")
    
    # ========== 步骤1: 选择 L1 Workflow Skill ==========
    print("\n" + "-" * 70)
    print("步骤1: 选择 L1 Workflow Skill")
    print("-" * 70)
    
    l1_skills = registry.get_by_level(SkillLevel.L1)
    print(f"\n可选的 L1 Skills ({len(l1_skills)} 个):")
    for skill in l1_skills:
        print(f"  - {skill.name}: {skill.description}")
    
    # 选择标准工作流
    l1_skill = registry.get("standard-workflow")
    if not l1_skill:
        print("错误: 未找到 standard-workflow")
        return
    
    print(f"\n✓ 选择: {l1_skill.name}")
    
    # ========== 步骤2: 选择 L2 Agent Skills ==========
    print("\n" + "-" * 70)
    print("步骤2: 选择 L2 Agent Skills")
    print("-" * 70)
    
    # 从 L1 的子 Skills 中选择
    selector = OperatorSkillSelector()
    l2_skills = selector.get_children_skills(l1_skill, all_skills)
    
    print(f"\n{l1_skill.name} 的子 Skills ({len(l2_skills)} 个):")
    for skill in l2_skills:
        print(f"  - {skill.name}: {skill.description[:50]}...")
    
    # 选择 coder-agent 和 verifier-agent
    selected_l2 = [
        registry.get("coder-agent"),
        registry.get("verifier-agent")
    ]
    selected_l2 = [s for s in selected_l2 if s]
    
    print(f"\n✓ 选择了 {len(selected_l2)} 个 L2 Skills:")
    for skill in selected_l2:
        print(f"  - {skill.name}")
    
    # ========== 步骤3: 选择 L3 DSL Skills ==========
    print("\n" + "-" * 70)
    print("步骤3: 选择 L3 DSL Skills")
    print("-" * 70)
    
    # 从 coder-agent 的子 Skills 中选择
    coder = registry.get("coder-agent")
    if coder:
        l3_candidates = selector.get_children_skills(coder, all_skills)
        print(f"\ncoder-agent 的子 Skills ({len(l3_candidates)} 个):")
        for skill in l3_candidates:
            print(f"  - {skill.name}: {skill.description[:50]}...")
        
        # 根据 DSL 类型筛选
        context = OperatorSelectionContext(
            operator_type=operator_type,
            backend=backend,
            dsl=dsl
        )
        
        l3_filtered = selector.coarse_filter(l3_candidates, context)
        
        if l3_filtered:
            print(f"\n✓ 根据 DSL={dsl} 筛选后: {len(l3_filtered)} 个")
            for skill in l3_filtered:
                print(f"  - {skill.name}")
        else:
            print(f"\n✗ 没有匹配 DSL={dsl} 的 L3 Skills，使用全部")
            l3_filtered = l3_candidates
    else:
        l3_filtered = []
    
    # ========== 步骤4: 拼接最终 prompt ==========
    print("\n" + "-" * 70)
    print("步骤4: 拼接最终 prompt")
    print("-" * 70)
    
    # 组合所有选中的 Skills
    selected_skills = [l1_skill] + selected_l2 + l3_filtered
    
    print(f"\n最终选中的 Skills ({len(selected_skills)} 个):")
    for skill in selected_skills:
        level_str = f"[{skill.level.value}]" if skill.level else "[--]"
        print(f"  {level_str} {skill.name}")
    
    # 构建 prompt
    task_description = f"""
请使用 Triton 为 {backend} 后端生成高性能的 {operator_type} 算子实现。

要求：
1. 使用 Triton 语法
2. 包含完整的 kernel 实现
3. 添加必要的注释
4. 考虑性能优化
"""
    
    final_prompt = build_prompt_with_skills(
        selected_skills,
        task_description,
        include_full_content=False  # 只包含描述，避免 prompt 过长
    )
    
    print(f"\n生成的 prompt 长度: {len(final_prompt)} 字符")
    print("\nprompt 预览（前 500 字符）:")
    print("-" * 70)
    print(final_prompt[:500])
    print("...")
    print("-" * 70)
    
    print()


async def example_4_with_history():
    """示例4：考虑执行历史的选择"""
    print("=" * 70)
    print("示例4：考虑执行历史的选择")
    print("=" * 70)
    
    if not HAS_LLM:
        print("[跳过] 需要 LLM 支持\n")
        return
    
    # 加载 Skills
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    # 创建选择器和 Agent
    selector = OperatorSkillSelector()
    agent = OperatorGenerationAgent()
    
    # 模拟执行历史
    history = [
        {
            "action": "使用 standard-workflow + coder-agent 生成代码",
            "result": "失败",
            "error": "性能不达标：只有 baseline 的 60%"
        },
        {
            "action": "使用 error-handling 进行优化",
            "result": "失败",
            "error": "仍然性能不足"
        }
    ]
    
    # 定义上下文
    context = OperatorSelectionContext(
        operator_type="matmul",
        backend="cuda",
        dsl="triton",
        custom_fields={"optimization_goal": "performance"}
    )
    
    print("\n任务上下文:")
    for k, v in context.to_dict().items():
        print(f"  {k}: {v}")
    
    print("\n执行历史:")
    for i, h in enumerate(history, 1):
        print(f"  {i}. {h['action']}")
        print(f"     结果: {h['result']}")
        if 'error' in h:
            print(f"     错误: {h['error']}")
    
    # 获取 L2 Agent Skills
    l2_skills = registry.get_by_level(SkillLevel.L2)
    
    # 构建带历史的 prompt（用户自定义模板，history 作为参数传递）
    prompt_template_with_history = """你是一个 Skill 选择专家。

任务上下文：
{context_str}

执行历史：
{history_info}

候选 Skills（L2层级）：
{skills_str}

基于执行历史，前面的尝试已经失败了。请选择更适合的 Skills 来解决问题。

返回 JSON 格式：
```json
{{
  "selected": ["skill-name-1", "skill-name-2"],
  "reason": "选择理由"
}}
```
"""
    # 格式化 history
    history_info = "\n".join([
        f"{i}. {h['action']}\n   结果: {h['result']}" + 
        (f"\n   错误: {h['error']}" if 'error' in h else "")
        for i, h in enumerate(history, 1)
    ])
    
    prompt = selector.build_llm_prompt(
        l2_skills, context, prompt_template_with_history,
        history_info=history_info
    )
    
    print(f"\n生成的 LLM prompt（前 800 字符）:")
    print("-" * 70)
    print(prompt[:800])
    print("...")
    print("-" * 70)
    
    # LLM 根据历史选择更合适的 Skills
    print("\n调用 LLM...")
    response = await agent.generate(prompt)
    
    print(f"\nLLM 响应预览（前 300 字符）:")
    print(response[:300])
    print("...")
    
    print()


def example_5_filter_by_backend():
    """示例5：按后端过滤"""
    print("=" * 70)
    print("示例5：按后端过滤")
    print("=" * 70)
    
    registry = SkillRegistry()
    skills_dir = Path(__file__).parent / "skills"
    registry.load_from_directory(skills_dir)
    
    selector = OperatorSkillSelector()
    
    # 测试不同后端
    backends = ["cuda", "ascend", "rocm"]
    
    for backend in backends:
        context = OperatorSelectionContext(
            backend=backend
        )
        
        all_skills = registry.get_all()
        filtered = selector.coarse_filter(all_skills, context)
        
        print(f"\n[Backend={backend}] 匹配的 Skills ({len(filtered)} 个):")
        for skill in filtered[:5]:  # 只显示前5个
            level_str = f"[{skill.level.value}]" if skill.level else "[--]"
            print(f"  {level_str} {skill.name}")
            if skill.metadata and 'backend' in skill.metadata:
                print(f"      支持: {skill.metadata['backend']}")
    
    print()


async def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("=" * 70)
    print("Skill System 算子生成场景示例")
    print("=" * 70)
    print()
    
    if HAS_LLM:
        print("LLM 配置信息:")
        print_settings_info()
        print()
    
    examples = [
        ("示例1: 基础算子选择", example_1_basic_operator_selection, False),
        ("示例2: LLM 精筛", example_2_llm_skill_selection, True),
        ("示例3: 完整流程", example_3_complete_workflow, True),
        ("示例4: 考虑历史", example_4_with_history, True),
        ("示例5: 按后端过滤", example_5_filter_by_backend, False),
    ]
    
    for name, example_func, is_async in examples:
        try:
            if is_async:
                await example_func()
            else:
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
    asyncio.run(run_all_examples())
