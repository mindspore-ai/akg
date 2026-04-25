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

"""测试 Workflow Registry 和 CoderOnlyWorkflow 集成"""

import pytest
from akg_agents.core_v2.workflows.registry import WorkflowRegistry, register_workflow


class TestWorkflowRegistry:
    """测试 WorkflowRegistry 基本功能"""
    
    def setup_method(self):
        """每个测试前清空注册表"""
        WorkflowRegistry.clear()
    
    def test_register_workflow_with_decorator(self):
        """测试使用装饰器注册 workflow"""
        @register_workflow(scopes=["test"])
        class MockWorkflow:
            TOOL_NAME = "use_mock_workflow"
            DESCRIPTION = "测试 workflow"
            PARAMETERS_SCHEMA = {"type": "object"}
        
        # 验证注册成功
        assert WorkflowRegistry.is_registered("MockWorkflow")
        assert "MockWorkflow" in WorkflowRegistry.list_workflows()
        
        # 验证 scope
        assert "MockWorkflow" in WorkflowRegistry.list_workflows(scope="test")
        assert "MockWorkflow" not in WorkflowRegistry.list_workflows(scope="op")
    
    def test_get_tool_config(self):
        """测试获取工具配置"""
        @register_workflow()
        class MockWorkflow:
            TOOL_NAME = "use_mock"
            DESCRIPTION = "测试描述"
            PARAMETERS_SCHEMA = {
                "type": "object",
                "properties": {
                    "param1": {"type": "string"}
                }
            }
        
        config = WorkflowRegistry.get_tool_config("MockWorkflow")
        
        assert config is not None
        assert "use_mock" in config
        assert config["use_mock"]["type"] == "call_workflow"
        assert config["use_mock"]["workflow_name"] == "MockWorkflow"
        assert config["use_mock"]["function"]["name"] == "use_mock"
        assert config["use_mock"]["function"]["description"] == "测试描述"
    
    def test_workflow_without_tool_config(self):
        """测试没有工具配置的 workflow"""
        @register_workflow()
        class IncompleteWorkflow:
            pass
        
        # 应该注册成功，但没有工具配置
        assert WorkflowRegistry.is_registered("IncompleteWorkflow")
        
        # 获取工具配置应该返回 None
        config = WorkflowRegistry.get_tool_config("IncompleteWorkflow")
        assert config is None


class TestCoderOnlyWorkflowIntegration:
    """测试 CoderOnlyWorkflow 的注册和集成"""
    
    def setup_method(self):
        """每个测试前清空注册表"""
        WorkflowRegistry.clear()
    
    def test_coder_only_workflow_registered(self):
        """测试 CoderOnlyWorkflow 是否正确注册"""
        from akg_agents.op.workflows.coder_only_workflow import CoderOnlyWorkflow

        # setup_method 调用了 clear()，Python import 缓存不会重新执行模块级装饰器，
        # 所以需要手动重新注册
        WorkflowRegistry.register(CoderOnlyWorkflow, scopes=["op"])
        
        # 验证注册成功
        assert WorkflowRegistry.is_registered("CoderOnlyWorkflow")
        assert "CoderOnlyWorkflow" in WorkflowRegistry.list_workflows(scope="op")
        
        # 验证工具配置
        config = WorkflowRegistry.get_tool_config("CoderOnlyWorkflow")
        assert config is not None
        assert "use_coder_only_workflow" in config
        
        tool_def = config["use_coder_only_workflow"]
        assert tool_def["type"] == "call_workflow"
        assert tool_def["workflow_name"] == "CoderOnlyWorkflow"
        
        # 验证参数 schema
        func = tool_def["function"]
        assert func["name"] == "use_coder_only_workflow"
        assert "CoderOnly workflow" in func["description"]
        
        params = func["parameters"]
        assert params["type"] == "object"
        assert "op_name" in params["properties"]
        assert "task_desc" in params["properties"]
        assert "dsl" in params["properties"]
        assert set(params["required"]) == {"op_name", "task_desc", "dsl", "framework", "backend", "arch"}
    
    def test_workflow_class_accessible(self):
        """测试可以获取 workflow 类"""
        from akg_agents.op.workflows.coder_only_workflow import CoderOnlyWorkflow
        
        # clear() 后需要重新注册（Python import 缓存不会重新触发装饰器）
        WorkflowRegistry.register(CoderOnlyWorkflow, scopes=["op"])
        
        workflow_class = WorkflowRegistry.get_workflow_class("CoderOnlyWorkflow")
        assert workflow_class is not None
        assert workflow_class == CoderOnlyWorkflow
        
        # 验证类属性
        assert hasattr(workflow_class, 'TOOL_NAME')
        assert hasattr(workflow_class, 'DESCRIPTION')
        assert hasattr(workflow_class, 'PARAMETERS_SCHEMA')
        assert hasattr(workflow_class, 'build_graph')


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
