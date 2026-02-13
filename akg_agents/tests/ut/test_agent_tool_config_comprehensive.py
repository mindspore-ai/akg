#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgentBase 工具配置加载功能 - 综合测试
"""

import sys
import os
import yaml

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from akg_agents.core_v2.agents.base import AgentBase


def test_1_basic_agent():
    """测试 1: 基本 Agent"""
    print("\n" + "=" * 80)
    print("测试 1: 基本 Agent - OpTaskBuilder")
    print("=" * 80)
    
    class OpTaskBuilderAgent(AgentBase):
        TOOL_NAME = "call_op_task_builder"
        DESCRIPTION = """将用户需求转换为 KernelBench 格式的 task 代码，并验证其可运行性。
这是算子生成的必要前置步骤，必须先确保 task 代码正确，才能调用后续的代码生成工具。"""
        PARAMETERS_SCHEMA = {
            "type": "object",
            "properties": {
                "op_name": {
                    "type": "string",
                    "description": "算子名称"
                },
                "user_request": {
                    "type": "string",
                    "description": "用户的自然语言需求描述"
                },
                "user_feedback": {
                    "type": "string",
                    "description": "用户对之前生成的 task_desc 的反馈（可选）",
                    "default": ""
                }
            },
            "required": ["op_name", "user_request"]
        }
    
    agent = OpTaskBuilderAgent()
    config = agent.load_tool_config()
    
    print("生成的配置字典:")
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False))
    print("✅ 测试通过\n")
    return True


def test_2_missing_metadata():
    """测试 2: 缺少元数据（预期失败）"""
    print("=" * 80)
    print("测试 2: 缺少元数据 - 应该抛出 ValueError")
    print("=" * 80)
    
    class IncompleteAgent(AgentBase):
        # 故意缺少所有必需字段
        pass
    
    agent = IncompleteAgent()
    
    try:
        config = agent.load_tool_config()
        print("❌ 测试失败：应该抛出 ValueError")
        return False
    except ValueError as e:
        print(f"✅ 成功捕获 ValueError:\n{e}\n")
        return True


def test_3_no_parameters():
    """测试 3: 无参数的 Agent"""
    print("=" * 80)
    print("测试 3: 无参数的 Agent")
    print("=" * 80)
    
    class NoParamAgent(AgentBase):
        TOOL_NAME = "call_no_param_agent"
        DESCRIPTION = "不需要参数的 Agent"
        PARAMETERS_SCHEMA = {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    agent = NoParamAgent()
    config = agent.load_tool_config()
    print("生成的配置字典:")
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False))
    print("✅ 测试通过\n")
    return True


def test_4_complex_parameters():
    """测试 4: 复杂参数类型"""
    print("=" * 80)
    print("测试 4: 复杂参数类型（包含数组、对象等）")
    print("=" * 80)
    
    class ComplexAgent(AgentBase):
        TOOL_NAME = "call_complex_agent"
        DESCRIPTION = "支持复杂参数类型的 Agent"
        PARAMETERS_SCHEMA = {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "名称"
                },
                "count": {
                    "type": "integer",
                    "description": "数量",
                    "default": 10
                },
                "ratio": {
                    "type": "number",
                    "description": "比例",
                    "default": 0.5
                },
                "enabled": {
                    "type": "boolean",
                    "description": "是否启用",
                    "default": True
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "标签列表"
                },
                "config": {
                    "type": "object",
                    "properties": {
                        "timeout": {"type": "integer"},
                        "retry": {"type": "boolean"}
                    },
                    "description": "配置对象"
                }
            },
            "required": ["name"]
        }
    
    agent = ComplexAgent()
    config = agent.load_tool_config()
    print("生成的配置字典:")
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False))
    print("✅ 测试通过\n")
    return True


def test_5_custom_indent():
    """测试 5: 字典格式输出"""
    print("=" * 80)
    print("测试 5: 字典格式输出")
    print("=" * 80)
    
    class CustomIndentAgent(AgentBase):
        TOOL_NAME = "call_custom_indent"
        DESCRIPTION = "测试字典格式"
        PARAMETERS_SCHEMA = {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "参数"}
            },
            "required": ["param"]
        }
    
    agent = CustomIndentAgent()
    config = agent.load_tool_config()
    
    print("生成的配置字典:")
    print(config)
    print("\n转换为 YAML (4 空格缩进):")
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False, indent=4))
    print("✅ 测试通过\n")
    return True


def test_6_multiline_description():
    """测试 6: 多行描述"""
    print("=" * 80)
    print("测试 6: 多行描述格式化")
    print("=" * 80)
    
    class MultilineAgent(AgentBase):
        TOOL_NAME = "call_multiline"
        DESCRIPTION = """这是第一行描述
这是第二行描述
这是第三行描述

这是空行后的描述"""
        PARAMETERS_SCHEMA = {
            "type": "object",
            "properties": {
                "input1": {"type": "string"},
                "input2": {"type": "integer"}
            },
            "required": ["input1", "input2"]
        }
    
    agent = MultilineAgent()
    config = agent.load_tool_config()
    print("生成的配置字典:")
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False))
    print("✅ 测试通过\n")
    return True


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("AgentBase.load_tool_config() - 综合功能测试")
    print("=" * 80)
    
    tests = [
        test_1_basic_agent,
        test_2_missing_metadata,
        test_3_no_parameters,
        test_4_complex_parameters,
        test_5_custom_indent,
        test_6_multiline_description,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("测试汇总")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("🎉 所有测试通过！")
        print("=" * 80)
        return 0
    else:
        print("❌ 部分测试失败")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
