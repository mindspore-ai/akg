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
Skill System 泛化性演示：智能邮件编写助手

演示 Skill System 的泛化能力：
1. Skill System 不仅限于算子生成
2. 通用的 SkillSelector 可用于任何领域
3. 从多个 Skills 中智能选择相关知识
4. 将专业知识注入 LLM prompt

运行方式：
    cd /path/to/akg_agents
    conda activate akg_agents
    source env.sh
    python examples/run_skill/07_demo_email_writing.py

注意：本示例需要配置 LLM
"""

from pathlib import Path
import asyncio

from akg_agents.core_v2.skill import (
    SkillInstaller, 
    SkillRegistry, 
    SkillSelector,
    SelectionContext
)

# 导入 LLM 相关
try:
    from akg_agents.core_v2.agents import AgentBase, Jinja2TemplateWrapper, register_agent
    HAS_LLM = True
except ImportError:
    print("[提示] 无法导入LLM模块\n")
    HAS_LLM = False
    import sys
    sys.exit(1)


@register_agent
class EmailWritingAgent(AgentBase):
    """邮件编写Agent"""
    
    def __init__(self):
        context = {"agent_name": "email_writing_agent"}
        super().__init__(context=context)
    
    async def generate(self, prompt: str, model_level: str = "standard"):
        """生成文本"""
        template = Jinja2TemplateWrapper("{{ prompt }}")
        
        content, formatted_prompt, reasoning_content = await self.run_llm(
            template, {"prompt": prompt}, model_level
        )
        
        return content


async def main(installer):
    """主函数
    
    Args:
        installer: SkillInstaller 实例
        
    Returns:
        本次安装的 skills 列表
    """
    print("\n" + "=" * 80)
    print("Skill System 泛化性演示：智能邮件编写助手")
    print("=" * 80)
    print()
    
    # 步骤1：安装和加载skills
    print("步骤1：准备Skills")
    print("-" * 80)
    
    skills_dir = Path(__file__).parent / "skills"
    
    print(f"安装skills...")
    install_results = installer.install_from_directory(skills_dir)
    installed_skills = [r['skill_name'] for r in install_results if 'error' not in r]
    
    registry = SkillRegistry()
    count = registry.load_from_directory(installer.install_root)
    print(f"[OK] 加载了 {count} 个skills\n")
    
    # 步骤2：使用LLM选择邮件skill
    print("步骤2：使用LLM选择邮件相关Skill")
    print("-" * 80)
    
    all_skills = registry.get_all()
    selector = SkillSelector()
    agent = EmailWritingAgent()
    
    context = SelectionContext(
        custom_fields={
            "task_type": "邮件编写",
            "framework": "专业商务邮件",
            "optimization_goal": "生成符合商务规范的专业邮件"
        }
    )
    
    print(f"从 {len(all_skills)} 个skills中检索...")
    print(f"任务上下文:")
    for k, v in context.to_dict().items():
        print(f"  {k}: {v}")
    
    # 构建prompt（用户自定义模板）
    prompt_template = """你是一个专业的 Skill 选择专家，负责为邮件编写任务选择合适的 Skill。

任务上下文：
{context_str}

可用的 Skills：
{skills_str}

请根据任务需求（专业商务邮件编写），选择最相关的Skill，不要有遗漏。

返回 JSON 格式：
```json
{{
  "selected": ["skill-name-1", "skill-name-2"],
  "reason": "选择理由"
}}
```
"""
    prompt = selector.build_llm_prompt(all_skills, context, prompt_template)
    
    # 调用LLM
    print("\n调用LLM...")
    llm_response = await agent.generate(prompt)
    
    # 解析响应
    selected = selector.parse_llm_response(llm_response, all_skills)
    
    if not selected:
        print("[错误] LLM未选中任何Skill")
        return
    
    print(f"\n[OK] LLM选择了 {len(selected)} 个skill:")
    for skill in selected:
        print(f"  - {skill.name}: {skill.description}")
    
    email_skill = selected[0]
    print()
    
    # 步骤3：提取skill内容
    print("步骤3：提取Skill内容")
    print("-" * 80)
    
    skill_content = email_skill.content
    print(f"[OK] 提取了 {len(skill_content)} 字符\n")
    
    # 步骤4：生成邮件
    print("步骤4：基于Skill知识生成邮件")
    print("-" * 80)
    
    user_request = """
写一封商务合作邮件：
收件人：李焱植（某AI科技公司CEO）
发件人：张汽水（TechCorp公司技术总监）
合作内容：共同研发AI平台、共享资源、联合推广
目标：建立合作意向，邀请会议讨论
语气：正式、专业
"""
    
    # 构造prompt（注意：大括号需要转义）
    def escape_braces(text: str) -> str:
        return text.replace('{', '{{').replace('}', '}}')
    
    escaped_content = escape_braces(skill_content)
    escaped_request = escape_braces(user_request)
    
    email_prompt = f"""你是专业的邮件助手。根据以下指南和需求生成邮件。

# 邮件编写指南
{escaped_content}

# 用户需求
{escaped_request}

# 任务
生成一封符合专业标准的邮件。

要求：
1. 主题清晰
2. 结构完整
3. 语气得体
4. 内容精准

请直接输出邮件内容。
"""
    
    print(f"构造prompt (总长度: {len(email_prompt)} 字符)")
    print("调用LLM生成邮件...\n")
    
    email = await agent.generate(email_prompt)
    
    print("=" * 80)
    print("生成的邮件：")
    print("=" * 80)
    print(email)
    print("=" * 80)
    print()
    
    print("[OK] 演示完成！")
    print()
    print("核心价值：")
    print("  1. Skill系统管理专业知识（邮件编写指南）")
    print("  2. LLM从多个skills中智能选择相关skill")
    print("  3. 将skill知识注入LLM prompt")
    print("  4. LLM基于专业知识生成标准邮件")
    print()
    print("泛化性体现：")
    print("  - Skill System 不仅限于算子生成")
    print("  - 通用 SkillSelector 可用于任何领域")
    print("  - 同一套架构支持多种应用场景")
    print()
    
    return installed_skills


async def main_with_cleanup():
    """带清理的主函数"""
    installer = SkillInstaller()
    installed_skills = []
    
    try:
        # 运行主逻辑并获取安装的skills列表
        installed_skills = await main(installer)
        
    finally:
        # 清理本次示例安装的skills
        if installed_skills:
            print("=" * 80)
            print("清理测试Skills")
            print("=" * 80)
            
            success_count = 0
            for skill_name in installed_skills:
                try:
                    if installer.uninstall(skill_name):
                        success_count += 1
                        print(f"  [OK] 卸载: {skill_name}")
                except Exception as e:
                    print(f"  [错误] 卸载 {skill_name} 失败: {e}")
            
            print(f"\n[OK] 清理完成: 成功卸载 {success_count}/{len(installed_skills)} 个skills")
            print()


if __name__ == "__main__":
    asyncio.run(main_with_cleanup())
