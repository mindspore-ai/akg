"""
示例5：Skill安装机制

演示如何使用SkillInstaller进行skill的安装管理：
1. 从项目目录安装skills到统一位置（~/.akg/skills/）
2. 基于哈希的增量更新（只复制/更新必要的文件）
3. 资源路径解析（让Agent/工具能可靠访问skill资源）
4. 安装信息查询和验证
"""

from pathlib import Path

from ai_kernel_generator.core_v2.skill import SkillInstaller, SkillRegistry, SkillMetadata


def example_1_basic_install():
    """示例1：基本安装"""
    print("=" * 60)
    print("示例1：基本安装")
    print("=" * 60)
    
    # 创建安装管理器
    installer = SkillInstaller()
    print(f"安装根目录: {installer.install_root}")
    print()
    
    # 安装单个skill
    skill_dir = Path(__file__).parent / "skills" / "triton-syntax"
    
    if skill_dir.exists():
        result = installer.install(skill_dir)
        print(f"[OK] 安装完成: {result['skill_name']}")
        print(f"     目标位置: {result['target_dir']}")
        print(f"     文件统计: 复制{result['stats']['copied']}个, "
              f"更新{result['stats']['updated']}个, "
              f"跳过{result['stats']['skipped']}个")
        print(f"     总文件数: {result['total_files']}")
    else:
        print(f"[跳过] skill目录不存在: {skill_dir}")
    
    print()


def example_2_batch_install():
    """示例2：批量安装"""
    print("=" * 60)
    print("示例2：批量安装")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    # 批量安装目录下所有skills
    skills_dir = Path(__file__).parent / "skills"
    
    if skills_dir.exists():
        results = installer.install_from_directory(skills_dir)
        
        print(f"[OK] 发现 {len(results)} 个skills")
        print()
        
        success_count = sum(1 for r in results if "error" not in r)
        print(f"[OK] 成功安装: {success_count}/{len(results)}")
        
        for result in results[:3]:  # 只显示前3个
            if "error" in result:
                print(f"     [失败] {result['skill_name']}: {result['error']}")
            else:
                stats = result['stats']
                print(f"     [OK] {result['skill_name']}: "
                      f"复制{stats['copied']}, 更新{stats['updated']}, 跳过{stats['skipped']}")
    else:
        print(f"[跳过] skills目录不存在: {skills_dir}")
    
    print()


def example_3_incremental_update():
    """示例3：增量更新（基于哈希）"""
    print("=" * 60)
    print("示例3：增量更新（基于哈希）")
    print("=" * 60)
    
    installer = SkillInstaller()
    skill_dir = Path(__file__).parent / "skills" / "triton-syntax"
    
    if not skill_dir.exists():
        print(f"[跳过] skill目录不存在: {skill_dir}")
        print()
        return
    
    # 第一次安装
    print("第一次安装:")
    result1 = installer.install(skill_dir)
    print(f"  复制: {result1['stats']['copied']}, "
          f"更新: {result1['stats']['updated']}, "
          f"跳过: {result1['stats']['skipped']}")
    print()
    
    # 第二次安装（所有文件应该被跳过）
    print("第二次安装（文件哈希相同）:")
    result2 = installer.install(skill_dir)
    print(f"  复制: {result2['stats']['copied']}, "
          f"更新: {result2['stats']['updated']}, "
          f"跳过: {result2['stats']['skipped']}")
    print()
    
    print("[OK] 增量更新机制：")
    print("     - 文件不存在 → 复制")
    print("     - 文件存在但哈希不同 → 覆盖")
    print("     - 文件存在且哈希相同 → 跳过")
    print()


def example_4_resolve_resource():
    """示例4：资源路径解析（核心功能）"""
    print("=" * 60)
    print("示例4：资源路径解析（供Agent/工具使用）")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    # 确保skill已安装
    skill_dir = Path(__file__).parent / "skills" / "triton-syntax"
    if skill_dir.exists():
        installer.install(skill_dir)
    
    # 解析资源路径
    skill_name = "triton-syntax"
    
    # 示例1：获取skill路径
    skill_path = installer.get_skill_path(skill_name)
    if skill_path:
        print(f"[OK] Skill路径: {skill_path}")
        print(f"     SKILL.md存在: {(skill_path / 'SKILL.md').exists()}")
        print()
    
    # 示例2：解析资源文件
    # 假设skill中有templates/kernel.triton文件
    resource_path = installer.resolve_resource(skill_name, "SKILL.md")
    if resource_path:
        print(f"[OK] 资源解析成功:")
        print(f"     {skill_name}/SKILL.md")
        print(f"     → {resource_path}")
        print(f"     文件存在: {resource_path.exists()}")
    else:
        print(f"[提示] 资源不存在或skill未安装")
    
    print()
    
    # 使用场景示例
    print("[使用场景] Agent如何使用:")
    print("  1. Agent接收任务：生成Triton kernel代码")
    print("  2. Agent查找skill：triton-syntax")
    print("  3. Agent需要模板：installer.resolve_resource('triton-syntax', 'templates/kernel.triton')")
    print("  4. 获得绝对路径：~/.akg/skills/triton-syntax/templates/kernel.triton")
    print("  5. 读取并使用模板")
    print()


def example_5_query_and_verify():
    """示例5：查询和验证"""
    print("=" * 60)
    print("示例5：查询和验证")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    # 列出所有已安装的skills
    installed = installer.list_installed()
    print(f"[OK] 已安装 {len(installed)} 个skills:")
    for skill_name, info in list(installed.items())[:5]:  # 只显示前5个
        print(f"     - {skill_name}")
        print(f"       安装时间: {info.get('installed_at', 'N/A')[:19]}")
        print(f"       文件数量: {info.get('file_count', 'N/A')}")
    print()
    
    # 检查特定skill是否安装
    test_skill = "triton-syntax"
    is_installed = installer.is_installed(test_skill)
    print(f"[OK] {test_skill} 安装状态: {'已安装' if is_installed else '未安装'}")
    print()
    
    # 验证完整性
    if is_installed:
        is_valid, error = installer.verify(test_skill)
        if is_valid:
            print(f"[OK] {test_skill} 验证通过")
        else:
            print(f"[警告] {test_skill} 验证失败: {error}")
        
        # 获取安装信息
        info = installer.get_install_info(test_skill)
        if info:
            print(f"     源路径: {info.get('source', 'N/A')}")
            print(f"     文件数: {len(info.get('files', {}))}")
    
    print()


def example_6_integrate_with_registry():
    """示例6：先安装后加载（正确的使用方式）"""
    print("=" * 60)
    print("示例6：先安装后加载（正确的使用方式）")
    print("=" * 60)
    
    # 步骤1：安装skills
    installer = SkillInstaller()
    skills_dir = Path(__file__).parent / "skills"
    
    if skills_dir.exists():
        print("[步骤1] 安装skills到 ~/.akg/skills/:")
        results = installer.install_from_directory(skills_dir)
        success_count = sum(1 for r in results if "error" not in r)
        print(f"  [OK] 成功安装 {success_count}/{len(results)} 个skills")
        print()
    
    # 步骤2：从安装位置加载
    print("[步骤2] 从 ~/.akg/skills/ 加载:")
    registry = SkillRegistry()
    
    # 方式1：直接加载整个安装目录
    count = registry.load_from_directory(installer.install_root)
    
    print(f"  [OK] 从安装位置加载了 {count} 个skills")
    print()
    
    # 查询已加载的skills
    all_skills = registry.get_all()
    print(f"[OK] Registry中共有 {len(all_skills)} 个skills:")
    for skill in list(all_skills)[:3]:
        print(f"     - {skill.name}")
        print(f"       路径: {skill.skill_path}")
    
    print()


def example_7_uninstall():
    """示例7：卸载skill"""
    print("=" * 60)
    print("示例7：卸载skill")
    print("=" * 60)
    
    installer = SkillInstaller()
    
    # 先安装一个skill
    skill_dir = Path(__file__).parent.parent / "skills" / "cuda-basics"
    if skill_dir.exists():
        result = installer.install(skill_dir)
        skill_name = result['skill_name']
        print(f"[OK] 安装skill: {skill_name}")
        
        # 检查是否安装
        print(f"[OK] 安装前检查: {installer.is_installed(skill_name)}")
        
        # 卸载
        success = installer.uninstall(skill_name)
        print(f"[OK] 卸载结果: {'成功' if success else '失败'}")
        
        # 再次检查
        print(f"[OK] 卸载后检查: {installer.is_installed(skill_name)}")
        
        # 重新安装回来
        installer.install(skill_dir)
        print(f"[OK] 重新安装: {skill_name}")
    else:
        print(f"[跳过] skill目录不存在: {skill_dir}")
    
    print()


def run_all_examples():
    """运行所有示例"""
    print("\n")
    print("=" * 60)
    print("Skill安装机制 - 完整示例")
    print("=" * 60)
    print()
    
    # 记录运行前已有的skills
    installer = SkillInstaller()
    before_install = set(installer.list_installed().keys())
    
    examples = [
        example_1_basic_install,
        example_2_batch_install,
        example_3_incremental_update,
        example_4_resolve_resource,
        example_5_query_and_verify,
        example_6_integrate_with_registry,
        example_7_uninstall,
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
