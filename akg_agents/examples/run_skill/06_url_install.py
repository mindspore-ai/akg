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
示例6：从URL安装skills

演示如何从GitHub等Git仓库直接安装skills：
1. 从GitHub安装单个skill
2. 从GitHub安装仓库中的特定skill
3. 使用便捷方法install_from_github
4. 处理安装错误
"""

from pathlib import Path

from akg_agents.core_v2.skill import SkillInstaller, SkillRegistry


def example_1_install_from_github():
    """示例1：从GitHub安装Vercel的skill"""
    print("=" * 60)
    print("示例1：从GitHub安装Vercel的skill")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    try:
        # 使用便捷方法从GitHub安装
        print("从GitHub安装 react-best-practices...")
        result = installer.install_from_github(
            repo="vercel-labs/agent-skills",
            skill_path="skills/react-best-practices"  # 注意：在skills/目录下
        )
        
        print(f"[OK] 安装成功: {result['skill_name']}")
        print(f"     目标位置: {result['target_dir']}")
        print(f"     文件统计: 复制{result['stats']['copied']}个, "
              f"更新{result['stats']['updated']}个, "
              f"跳过{result['stats']['skipped']}个")
        print(f"     总文件数: {result['total_files']}")
        print(f"     源URL: {result['source_url']}")
        
    except Exception as e:
        print(f"[错误] 安装失败: {e}")
        print("     提示：需要安装git命令行工具")
    
    print()


def example_2_install_from_url():
    """示例2：从完整URL安装"""
    print("=" * 60)
    print("示例2：从完整URL安装")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    try:
        # 使用完整URL安装
        print("从URL安装skill...")
        result = installer.install_from_url(
            url="https://github.com/vercel-labs/agent-skills.git",
            skill_path="skills/react-best-practices",  # 完整路径
            branch="main"
        )
        
        print(f"[OK] 安装成功: {result['skill_name']}")
        print(f"     从: {result['source_url']}")
        print(f"     路径: {result.get('source_path', 'N/A')}")
        
    except Exception as e:
        print(f"[错误] 安装失败: {e}")
    
    print()


def example_3_verify_installed():
    """示例3：验证已安装的skill"""
    print("=" * 60)
    print("示例3：验证已安装的skill")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    skill_name = "vercel-react-best-practices"
    
    # 检查是否安装
    if installer.is_installed(skill_name):
        print(f"[OK] {skill_name} 已安装")
        
        # 获取安装路径
        skill_path = installer.get_skill_path(skill_name)
        print(f"     安装路径: {skill_path}")
        
        # 验证完整性
        is_valid, error = installer.verify(skill_name)
        if is_valid:
            print(f"     验证状态: 通过")
        else:
            print(f"     验证状态: 失败 - {error}")
        
        # 获取安装信息
        info = installer.get_install_info(skill_name)
        if info:
            print(f"     版本: {info.get('version', 'N/A')}")
            print(f"     安装时间: {info.get('installed_at', 'N/A')[:19]}")
            print(f"     文件数: {len(info.get('files', {}))}")
            print(f"     源路径: {info.get('source', 'N/A')}")
    else:
        print(f"[提示] {skill_name} 未安装")
    
    print()


def example_4_use_installed_skill():
    """示例4：使用已安装的skill"""
    print("=" * 60)
    print("示例4：使用已安装的skill")
    print("=" * 60)
    
    installer = SkillInstaller()
    registry = SkillRegistry()
    
    # 从安装位置加载
    count = registry.load_from_directory(installer.install_root)
    print(f"[OK] 从安装位置加载了 {count} 个skills")
    print()
    
    # 查找Vercel的skill（使用SKILL.md中定义的名称）
    skill = registry.get("vercel-react-best-practices")
    if skill:
        print(f"[OK] 找到skill: {skill.name}")
        print(f"     描述: {skill.description[:80]}...")
        print(f"     路径: {skill.skill_path}")
        print()
        
        # 检查rules目录
        skill_dir = installer.get_skill_path(skill.name)
        if skill_dir:
            rules_dir = skill_dir / "rules"
            if rules_dir.exists():
                rules = list(rules_dir.glob("*.md"))
                print(f"[OK] 包含 {len(rules)} 个规则文件")
                if rules:
                    print("     前3个规则:")
                    for rule in rules[:3]:
                        print(f"       - {rule.name}")
    else:
        print("[提示] 未找到skill，请先运行示例1安装")
    
    print()


def example_5_resource_resolution():
    """示例5：解析Vercel skill的资源"""
    print("=" * 60)
    print("示例5：解析Vercel skill的资源")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    skill_name = "vercel-react-best-practices"
    
    if not installer.is_installed(skill_name):
        print(f"[提示] {skill_name} 未安装")
        print()
        return
    
    # 解析AGENTS.md
    agents_md = installer.resolve_resource(skill_name, "AGENTS.md")
    if agents_md and agents_md.exists():
        print(f"[OK] 找到AGENTS.md:")
        print(f"     路径: {agents_md}")
        print(f"     大小: {agents_md.stat().st_size} 字节")
    
    # 解析rules目录
    skill_path = installer.get_skill_path(skill_name)
    if skill_path:
        rules_dir = skill_path / "rules"
        if rules_dir.exists():
            print(f"\n[OK] Rules目录:")
            print(f"     路径: {rules_dir}")
            
            # 示例：读取一个rule
            async_parallel = rules_dir / "async-parallel.md"
            if async_parallel.exists():
                print(f"\n[示例] 读取规则文件:")
                print(f"     文件: async-parallel.md")
                content = async_parallel.read_text(encoding='utf-8')
                print(f"     内容长度: {len(content)} 字符")
                # 显示前200个字符
                print(f"     内容预览:")
                print(f"     {content[:200]}...")
    
    print()


def example_6_error_handling():
    """示例6：错误处理"""
    print("=" * 60)
    print("示例6：错误处理")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    # 测试1：无效的仓库
    print("[测试1] 安装不存在的仓库:")
    try:
        installer.install_from_github(
            repo="invalid-user/invalid-repo",
            skill_path="some-skill"
        )
    except Exception as e:
        print(f"  [预期错误] {type(e).__name__}: {str(e)[:100]}")
    print()
    
    # 测试2：仓库存在但路径不存在
    print("[测试2] 安装不存在的skill路径:")
    try:
        installer.install_from_github(
            repo="vercel-labs/agent-skills",
            skill_path="skills/non-existent-skill"
        )
    except Exception as e:
        print(f"  [预期错误] {type(e).__name__}: {str(e)[:100]}")
    print()
    
    # 测试3：没有SKILL.md的目录
    print("[测试3] 说明:")
    print("  如果目标目录没有SKILL.md，安装会失败")
    print("  这确保了只安装有效的skills")
    print()


def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("=" * 60)
    print("从URL安装Skills - 完整示例")
    print("=" * 60)
    print()
    
    print("注意：需要安装git命令行工具")
    print("Windows: https://git-scm.com/download/win")
    print("Linux: sudo apt-get install git")
    print("Mac: brew install git")
    print()
    
    # 记录运行前已有的skills
    installer = SkillInstaller()
    before_install = set(installer.list_installed().keys())
    
    examples = [
        example_1_install_from_github,
        example_2_install_from_url,
        example_3_verify_installed,
        example_4_use_installed_skill,
        example_5_resource_resolution,
        example_6_error_handling,
    ]
    
    try:
        for example in examples:
            try:
                example()
            except Exception as e:
                print(f"[错误] {example.__name__}: {e}")
                import traceback
                traceback.print_exc()
                print()
        
        print("=" * 60)
        print("所有示例运行完成")
        print("=" * 60)
        print()
    
    finally:
        # 清理本次示例安装的skills（增量清理）
        after_install = set(installer.list_installed().keys())
        new_skills = list(after_install - before_install)
        
        if new_skills:
            print("=" * 60)
            print("清理本次示例安装的skills")
            print("=" * 60)
            
            success_count = 0
            for skill_name in new_skills:
                try:
                    if installer.uninstall(skill_name):
                        success_count += 1
                        print(f"  [OK] 卸载: {skill_name}")
                except Exception as e:
                    print(f"  [错误] 卸载 {skill_name} 失败: {e}")
            
            print(f"\n[OK] 清理完成: 成功卸载 {success_count}/{len(new_skills)} 个skills")
            print()
        else:
            print("\n[OK] 没有需要清理的新安装skills\n")


if __name__ == "__main__":
    run_all_examples()
