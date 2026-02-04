#!/usr/bin/env python3
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
测试所有 Workflow 作为工具被 KernelAgent 调用

测试的 Workflows：
- CoderOnlyWorkflow: 跳过设计，直接生成+验证
- DefaultWorkflow: 完整流程（设计→生成→验证）
- VerifierOnlyWorkflow: 仅验证已有代码
- ConnectAllWorkflow: 灵活流程，AI 智能决策

运行方式：
  python examples/test_coder_only_workflow_tool.py
"""

import asyncio
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

logger = logging.getLogger(__name__)


async def test_workflow_registration():
    """测试 1: 验证 workflow 注册"""
    print("\n" + "="*80)
    print("测试 1: 验证 Workflow 注册")
    print("="*80)
    
    from akg_agents.core_v2.workflows.registry import WorkflowRegistry
    from akg_agents.op.workflows.coder_only_workflow import CoderOnlyWorkflow
    from akg_agents.op.workflows.default_workflow import DefaultWorkflow
    from akg_agents.op.workflows.verifier_only_workflow import VerifierOnlyWorkflow
    from akg_agents.op.workflows.connect_all_workflow import ConnectAllWorkflow
    
    # 检查是否注册
    workflows = WorkflowRegistry.list_workflows(scope="op")
    print(f"✓ 已注册的 workflows ({len(workflows)}): {workflows}")
    
    expected_workflows = ["CoderOnlyWorkflow", "DefaultWorkflow", "VerifierOnlyWorkflow", "ConnectAllWorkflow"]
    missing = [w for w in expected_workflows if w not in workflows]
    
    if missing:
        print(f"✗ 缺少 workflows: {missing}")
        return False
    
    # 检查工具配置
    expected_tools = {
        "CoderOnlyWorkflow": "use_coder_only_workflow",
        "DefaultWorkflow": "use_default_workflow",
        "VerifierOnlyWorkflow": "use_verifier_only_workflow",
        "ConnectAllWorkflow": "use_connect_all_workflow"
    }
    
    for workflow_name, tool_name in expected_tools.items():
        config = WorkflowRegistry.get_tool_config(workflow_name)
        if not config:
            print(f"✗ {workflow_name} 没有工具配置！")
            return False
        if tool_name not in config:
            print(f"✗ {workflow_name} 的工具名称不匹配！")
            return False
        print(f"✓ {workflow_name} → {tool_name}")
    
    return True


async def test_kernel_agent_loading():
    """测试 2: 验证 KernelAgent 加载 workflow"""
    print("\n" + "="*80)
    print("测试 2: KernelAgent 加载 Workflow")
    print("="*80)
    
    try:
        from akg_agents.op.agents.kernel_agent import KernelAgent
        
        # 创建 KernelAgent
        agent = KernelAgent(
            task_id="test_workflow_tool",
            model_level="fast",
            framework="torch",
            backend="cpu",
            arch="x86_64",
            dsl="cpp"
        )
        
        # 检查 workflow_registry
        print(f"✓ Workflow registry 大小: {len(agent.workflow_registry)}")
        print(f"✓ 已注册的 workflow 工具: {list(agent.workflow_registry.keys())}")
        
        # 检查 available_tools 中是否包含 workflow
        workflow_tools = [
            t for t in agent.available_tools 
            if t.get("function", {}).get("name") == "use_coder_only_workflow"
        ]
        
        if not workflow_tools:
            print("✗ available_tools 中没有找到 use_coder_only_workflow！")
            return False
        
        print(f"✓ 在 available_tools 中找到 workflow 工具")
        print(f"  描述: {workflow_tools[0]['function']['description'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ 创建 KernelAgent 失败: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_workflow_tool_info():
    """测试 3: 查看 workflow 工具的完整信息"""
    print("\n" + "="*80)
    print("测试 3: Workflow 工具详细信息")
    print("="*80)
    
    from akg_agents.core_v2.workflows.registry import WorkflowRegistry
    
    workflows = ["CoderOnlyWorkflow", "DefaultWorkflow", "VerifierOnlyWorkflow", "ConnectAllWorkflow"]
    
    for workflow_name in workflows:
        config = WorkflowRegistry.get_tool_config(workflow_name)
        if not config:
            continue
        
        tool_name = list(config.keys())[0]
        tool_def = config[tool_name]
        
        print(f"\n{'='*80}")
        print(f"Workflow: {workflow_name}")
        print(f"工具名称: {tool_def['function']['name']}")
        print(f"工具类型: {tool_def['type']}")
        desc_lines = tool_def['function']['description'].strip().split('\n')
        print(f"描述: {desc_lines[0]}")
        print(f"必需参数: {tool_def['function']['parameters']['required']}")
    
    return True


async def main():
    """主测试流程"""
    print("\n" + "="*80)
    print("所有 Workflow 工具化集成测试")
    print("="*80)
    
    tests = [
        ("Workflow 注册", test_workflow_registration),
        ("KernelAgent 加载", test_kernel_agent_loading),
        ("工具详细信息", test_workflow_tool_info),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{test_name}' 异常: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 汇总结果
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status} - {test_name}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！所有 Workflow 已成功集成为工具。")
        print("   - CoderOnlyWorkflow")
        print("   - DefaultWorkflow")
        print("   - VerifierOnlyWorkflow")
        print("   - ConnectAllWorkflow")
    else:
        print("\n⚠️  部分测试失败，请检查上面的错误信息。")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
