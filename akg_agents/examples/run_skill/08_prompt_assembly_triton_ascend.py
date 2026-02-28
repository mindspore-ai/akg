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
Triton-Ascend Prompt 拼接示例 - ReLU 和 MatMul

演示如何使用 Skill System 为不同算子生成完整的 Prompt：
1. 使用 metadata (backend, dsl) 进行初筛
2. 调用真实 LLM 选择导入哪些文档
3. 按照 level → category → name 顺序拼接 prompt

任务输入来自 KernelBench：
- ReLU: level1/19_ReLU.py
- MatMul: level1/1_Square_matrix_multiplication_.py

运行方式：
    cd /path/to/akg_agents
    conda activate akg_agents
    source env.sh
    python examples/run_skill/10_prompt_assembly_triton_ascend.py
"""

import asyncio
from pathlib import Path
from typing import List
import json

from akg_agents.core_v2.skill import SkillRegistry, SkillMetadata
from akg_agents.op.skill import OperatorSkillSelector, OperatorSelectionContext
from akg_agents.core_v2.agents import AgentBase, Jinja2TemplateWrapper, register_agent
from akg_agents.core_v2.config import print_settings_info


# Category 顺序
CATEGORY_ORDER = ["fundamental", "method", "implementation", "example"]


@register_agent
class PromptAssemblerAgent(AgentBase):
    """Prompt 拼接 Agent"""
    
    def __init__(self):
        context = {"agent_name": "PromptAssemblerAgent"}
        super().__init__(context=context)


class PromptAssembler:
    """Prompt 拼接器"""
    
    def __init__(self, skills_base_dir: Path):
        self.registry = SkillRegistry()
        self.skills_base_dir = skills_base_dir
        self.agent = PromptAssemblerAgent()
        self.selector = OperatorSkillSelector()  # 使用定义好的 selector
        
    def load_triton_ascend_skills(self):
        """加载 Triton-Ascend Skills"""
        triton_ascend_dir = self.skills_base_dir / "triton-ascend"
        count = self.registry.load_from_directory(triton_ascend_dir)
        print(f"✅ 成功加载 {count} 个 Triton-Ascend Skills\n")
        return count
    
    def initial_filter(self, backend: str, dsl: str) -> List[SkillMetadata]:
        """阶段1: 使用 metadata 的 backend 和 dsl 进行初筛"""
        print("=" * 70)
        print("阶段1: 初筛 - 使用 backend + dsl 过滤")
        print("=" * 70)
        print(f"筛选条件: backend={backend}, dsl={dsl}\n")
        
        # 使用 OperatorSelectionContext 和 selector 进行过滤
        context = OperatorSelectionContext(
            backend=backend,
            dsl=dsl
        )
        
        all_skills = self.registry.get_all()
        filtered = self.selector.coarse_filter(all_skills, context)
        
        print(f"初筛结果: {len(filtered)} 个 Skills")
        for skill in filtered:
            print(f"  - {skill.name} (Category: {skill.category or 'N/A'})")
        print()
        
        return filtered
    
    async def llm_select_skills(
        self, 
        filtered_skills: List[SkillMetadata],
        operator_type: str,
        operator_description: str,
        framework: str
    ) -> List[str]:
        """阶段2: 调用真实 LLM 选择导入哪些文档"""
        print("=" * 70)
        print("阶段2: LLM 选择 - 决定导入哪些文档")
        print("=" * 70)
        print(f"算子类型: {operator_type}")
        print(f"算子描述: {operator_description}")
        print(f"框架: {framework}\n")
        
        # 构建 Skills 信息列表
        skills_info = []
        for skill in filtered_skills:
            info = {
                "name": skill.name,
                "description": skill.description,
                "category": skill.category or "N/A",
                "operator_patterns": skill.metadata.get("operator_patterns", ""),
                "algorithms": skill.metadata.get("algorithms", ""),
                "framework": skill.metadata.get("framework", ""),
            }
            skills_info.append(info)
        
        # 简化的 LLM prompt，强调完备性
        llm_prompt = f"""你是一个 Triton Ascend 知识管理专家。现在需要为一个算子生成代码，请从以下 Skills 中选择相关的文档。

**算子信息**:
- 算子类型: {operator_type}
- 算子描述: {operator_description}
- 目标框架: {framework}

**可用的 Skills**:
{json.dumps(skills_info, indent=2, ensure_ascii=False)}

**任务**:
请选择足够相关的 Skills，确保不要漏掉任何有用的文档。返回一个 JSON 数组，只包含选中的 Skill 名称。

示例输出格式:
["triton-ascend-basics", "triton-ascend-api", ...]
"""
        
        print("调用 LLM 进行选择...")
        
        # 使用 AgentBase 的 run_llm 方法
        template = Jinja2TemplateWrapper("{{ prompt }}")
        content, formatted_prompt, reasoning_content = await self.agent.run_llm(
            template, 
            {"prompt": llm_prompt}, 
            "standard"
        )
        
        # 解析 LLM 返回的 JSON
        try:
            selected_names = json.loads(content)
        except json.JSONDecodeError:
            # 如果无法解析，尝试提取 JSON 数组
            import re
            match = re.search(r'\[.*?\]', content, re.DOTALL)
            if match:
                selected_names = json.loads(match.group())
            else:
                print("⚠️ 无法解析 LLM 响应，使用所有基础 Skills")
                selected_names = [s.name for s in filtered_skills if (s.category or "") in ("guide", "fundamental", "dsl")]
        
        print("LLM 选择结果:")
        for name in selected_names:
            skill = self.registry.get(name)
            if skill:
                category = skill.category or "N/A"
                print(f"  ✓ {name}")
                print(f"    Category: {category}")
            else:
                print(f"  ⚠️ {name} (未找到)")
        print()
        
        return selected_names
    
    def assemble_prompt(
        self,
        selected_skill_names: List[str],
        operator_name: str,
        task_description: str
    ) -> str:
        """阶段3: 按照 category → name 顺序拼接 prompt"""
        print("=" * 70)
        print("阶段3: Prompt 拼接 - 按 category → name 排序")
        print("=" * 70)
        
        # 获取 Skills
        skills_to_sort = []
        for name in selected_skill_names:
            skill = self.registry.get(name)
            if skill:
                skills_to_sort.append(skill)
        
        # 排序：category（按 CATEGORY_ORDER） → name
        def sort_key(skill: SkillMetadata):
            try:
                category_idx = CATEGORY_ORDER.index(skill.category or "")
            except ValueError:
                category_idx = 999
            return (category_idx, skill.name)
        
        sorted_skills = sorted(skills_to_sort, key=sort_key)
        
        print("拼接顺序:")
        for idx, skill in enumerate(sorted_skills, 1):
            category = skill.category or "N/A"
            print(f"  {idx}. {skill.name}")
            print(f"     [{category}]")
        print()
        
        # 拼接 Prompt（只有一段 System Prompt + 所有 Skill 内容）
        system_prompt = f"""你是一个专业的 Triton Ascend Kernel 编写专家。
当前任务是为 **{operator_name}** 算子生成高效的 Triton Ascend Kernel 代码。

以下是相关的技术文档和指南：
"""
        
        # 拼接所有 Skill 内容
        skill_contents = "\n\n---\n\n".join([skill.content for skill in sorted_skills])
        
        # 任务描述
        task_section = f"""
---

## 当前任务

### 算子名称
{operator_name}

### 任务描述
{task_description}

### 输出要求
请生成完整的 Triton Ascend Kernel 代码，包括：
1. Kernel 函数定义（使用 @triton.jit 装饰器）
2. ModelNew 类（继承 torch.nn.Module）
3. forward 方法中正确调用 kernel
4. 所有必要的 import 语句
"""
        
        final_prompt = f"{system_prompt}\n\n{skill_contents}\n\n{task_section}"
        
        return final_prompt


async def test_relu():
    """测试 ReLU 算子的 Prompt 拼接"""
    
    print("\n" + "=" * 70)
    print("测试案例 1: ReLU (Element-wise)")
    print("=" * 70)
    print()
    
    # 任务输入来自 KernelBench level1/19_ReLU.py
    operator_type = "elementwise"
    operator_name = "ReLU"
    operator_description = """
ReLU 激活函数实现。

**参考实现**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)
```

**输入**:
- x: 输入张量，shape 为 (batch_size, dim)，如 (16, 16384)
- dtype: float16 或 float32

**输出**:
- y: 输出张量，shape 与输入相同
- 计算公式: y = max(0, x)

**性能要求**:
1. 支持大 batch_size 和 dim
2. 内存访问优化（连续访问、256B 对齐）
3. 使用 VEC 核心
4. BLOCK_SIZE = 1024
"""
    
    framework = "torch"
    backend = "ascend"
    dsl = "triton-ascend"
    
    # 初始化
    skills_dir = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "resources" / "skills"
    assembler = PromptAssembler(skills_dir)
    
    # 加载 Skills
    assembler.load_triton_ascend_skills()
    
    # 阶段1: 初筛
    filtered_skills = assembler.initial_filter(backend, dsl)
    
    # 阶段2: LLM 选择
    selected_names = await assembler.llm_select_skills(
        filtered_skills,
        operator_type,
        operator_description,
        framework
    )
    
    # 阶段3: 拼接 Prompt
    final_prompt = assembler.assemble_prompt(
        selected_names,
        operator_name,
        operator_description
    )
    
    # 输出结果
    print("=" * 70)
    print("最终生成的 Prompt")
    print("=" * 70)
    print(f"总长度: {len(final_prompt)} 字符")
    print(f"总行数: {final_prompt.count(chr(10)) + 1} 行")
    print()
    
    # 保存到文件
    output_file = Path(__file__).parent / "output_relu_prompt.txt"
    output_file.write_text(final_prompt, encoding="utf-8")
    print(f"✅ Prompt 已保存到: {output_file}")
    print()
    
    # 显示前500字符预览
    print("Prompt 预览（前500字符）:")
    print("-" * 70)
    print(final_prompt[:500])
    print("...")
    print("-" * 70)
    print()
    
    print("✅ ReLU Prompt 拼接完成！")
    print()


async def test_matmul():
    """测试 MatMul 算子的 Prompt 拼接"""
    
    print("\n" + "=" * 70)
    print("测试案例 2: MatMul (矩阵乘法)")
    print("=" * 70)
    print()
    
    # 任务输入来自 KernelBench level1/1_Square_matrix_multiplication_.py
    operator_type = "matmul"
    operator_name = "MatMul"
    operator_description = """
方阵矩阵乘法实现。

**参考实现**:
```python
def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.matmul(A, B)
```

**输入**:
- A: 左矩阵，shape 为 (N, N)，如 (2048, 2048)
- B: 右矩阵，shape 为 (N, N)
- dtype: float16 或 float32

**输出**:
- C: 输出矩阵，shape 为 (N, N)
- 计算公式: C = A @ B

**性能要求**:
1. 使用 `tl.dot` 进行矩阵乘法计算
2. 使用 `tl.make_block_ptr` 和 `tl.advance` 优化内存访问
3. 使用固定 CUBE 核心数启动
4. **关键**: Ascend 后端的切分要求（BLOCK_M * BLOCK_K 对齐到 512Bytes）
5. BLOCK_M = 128, BLOCK_N = 128, BLOCK_K = 32
6. 支持大矩阵（N 可能很大）
7. 使用交错循环处理超大矩阵
"""
    
    framework = "torch"
    backend = "ascend"
    dsl = "triton-ascend"
    
    # 初始化
    skills_dir = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "resources" / "skills"
    assembler = PromptAssembler(skills_dir)
    
    # 加载 Skills
    assembler.load_triton_ascend_skills()
    
    # 阶段1: 初筛
    filtered_skills = assembler.initial_filter(backend, dsl)
    
    # 阶段2: LLM 选择
    selected_names = await assembler.llm_select_skills(
        filtered_skills,
        operator_type,
        operator_description,
        framework
    )
    
    # 阶段3: 拼接 Prompt
    final_prompt = assembler.assemble_prompt(
        selected_names,
        operator_name,
        operator_description
    )
    
    # 输出结果
    print("=" * 70)
    print("最终生成的 Prompt")
    print("=" * 70)
    print(f"总长度: {len(final_prompt)} 字符")
    print(f"总行数: {final_prompt.count(chr(10)) + 1} 行")
    print()
    
    # 保存到文件
    output_file = Path(__file__).parent / "output_matmul_prompt.txt"
    output_file.write_text(final_prompt, encoding="utf-8")
    print(f"✅ Prompt 已保存到: {output_file}")
    print()
    
    # 显示前500字符预览
    print("Prompt 预览（前500字符）:")
    print("-" * 70)
    print(final_prompt[:500])
    print("...")
    print("-" * 70)
    print()
    
    print("✅ MatMul Prompt 拼接完成！")
    print()


async def main():
    """主函数：测试 ReLU 和 MatMul"""
    
    print("\n" + "=" * 70)
    print("Triton-Ascend Prompt 拼接测试")
    print("=" * 70)
    print()
    
    # 打印配置信息
    print_settings_info()
    print()
    
    try:
        # 测试 ReLU
        await test_relu()
        
        # 测试 MatMul
        await test_matmul()
        
        print("=" * 70)
        print("所有测试完成！")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    asyncio.run(main())
