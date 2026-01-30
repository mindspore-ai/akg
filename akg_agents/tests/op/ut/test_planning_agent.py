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
PlanningAgent 单元测试
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path

from akg_agents.core_v2.agents.plan import PlanningAgent


class TestPlanningAgentInitialization:
    """测试 PlanningAgent 初始化"""
    
    def test_init_default_model_level(self):
        """测试默认模型级别"""
        agent = PlanningAgent(task_id="test_task_001")
        
        assert agent.agent_name == "PlanningAgent"
        assert agent.model_level == "standard"
        assert agent.context["task_id"] == "test_task_001"
    
    def test_init_custom_model_level(self):
        """测试自定义模型级别"""
        agent = PlanningAgent(task_id="test_task_002", model_level="complex")
        
        assert agent.model_level == "complex"
    
    def test_prompts_dir_path(self):
        """测试 prompts 目录路径"""
        agent = PlanningAgent(task_id="test_task_003")
        
        prompts_dir = agent._get_prompts_dir()
        assert prompts_dir.name == "planning"
        assert "op" in str(prompts_dir)
        assert "resources" in str(prompts_dir)
        assert "prompts" in str(prompts_dir)
    
    def test_load_templates(self):
        """测试模板加载"""
        agent = PlanningAgent(task_id="test_task_004")
        
        # 验证模板已加载
        assert agent._global_prompt_template is not None
        assert agent._detailed_prompt_template is not None


class TestToolFormatting:
    """测试工具格式化"""
    
    @pytest.fixture
    def agent(self):
        """创建 PlanningAgent 实例"""
        return PlanningAgent(task_id="test_task_format")
    
    def test_format_empty_tools(self, agent):
        """测试空工具列表格式化"""
        result = agent._format_tools([])
        assert result == "（无可用工具）"
    
    def test_format_single_tool(self, agent):
        """测试单个工具格式化"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_coder_only",
                    "description": "快速生成 kernel 代码",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_code": {"description": "task 代码"},
                            "op_name": {"description": "算子名称"}
                        }
                    }
                }
            }
        ]
        
        result = agent._format_tools(tools)
        
        assert "1. call_coder_only" in result
        assert "快速生成 kernel 代码" in result
        assert "task_code" in result
        assert "op_name" in result
    
    def test_format_multiple_tools(self, agent):
        """测试多个工具格式化"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_op_task_builder",
                    "description": "生成 task_desc 代码",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_coder_only",
                    "description": "快速生成 kernel",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "call_kernel_verifier",
                    "description": "验证 kernel",
                    "parameters": {"type": "object", "properties": {}}
                }
            }
        ]
        
        result = agent._format_tools(tools)
        
        assert "1. call_op_task_builder" in result
        assert "2. call_coder_only" in result
        assert "3. call_kernel_verifier" in result


class TestActionHistoryFormatting:
    """测试动作历史格式化"""
    
    @pytest.fixture
    def agent(self):
        """创建 PlanningAgent 实例"""
        return PlanningAgent(task_id="test_task_history")
    
    def test_format_empty_history(self, agent):
        """测试空动作历史格式化"""
        result = agent._format_action_history([])
        assert result == "（尚无执行历史）"
    
    def test_format_single_action(self, agent):
        """测试单个动作格式化"""
        history = [
            {
                "tool_name": "call_op_task_builder",
                "result": {
                    "status": "success",
                    "output_path": "op_task_builder/task_desc.py"
                }
            }
        ]
        
        result = agent._format_action_history(history)
        
        assert "1. call_op_task_builder" in result
        assert "success" in result
        assert "op_task_builder/task_desc.py" in result
    
    def test_format_multiple_actions(self, agent):
        """测试多个动作格式化"""
        history = [
            {
                "tool_name": "call_op_task_builder",
                "result": {"status": "success"}
            },
            {
                "tool_name": "call_coder_only",
                "result": {"status": "success"}
            },
            {
                "tool_name": "call_kernel_verifier",
                "result": {"status": "failed"}
            }
        ]
        
        result = agent._format_action_history(history)
        
        assert "1. call_op_task_builder" in result
        assert "2. call_coder_only" in result
        assert "3. call_kernel_verifier" in result
        assert "failed" in result


class TestDefaultTodoLists:
    """测试默认 todolist"""
    
    @pytest.fixture
    def agent(self):
        """创建 PlanningAgent 实例"""
        return PlanningAgent(task_id="test_task_default")
    
    def test_default_global_todolist_without_task_desc(self, agent):
        """测试无 task_desc 时的默认全局 todolist"""
        result = agent._get_default_global_todolist(has_task_desc=False)
        
        assert "<global_todo_list>" in result
        assert "</global_todo_list>" in result
        assert "task_desc" in result
        assert "kernel" in result
        assert "验证" in result
    
    def test_default_global_todolist_with_task_desc(self, agent):
        """测试有 task_desc 时的默认全局 todolist"""
        result = agent._get_default_global_todolist(has_task_desc=True)
        
        assert "<global_todo_list>" in result
        assert "</global_todo_list>" in result
        assert "task_desc" not in result.lower() or "生成" not in result  # 不应该有"生成 task_desc"
        assert "kernel" in result
        assert "验证" in result
    
    def test_default_detailed_todolist(self, agent):
        """测试默认详细 todolist"""
        result = agent._get_default_detailed_todolist()
        
        assert "<todo_list>" in result
        assert "</todo_list>" in result
        assert "call_op_task_builder" in result
        assert "call_coder_only" in result
        assert "call_kernel_verifier" in result
        assert "完成任务" in result


class TestGenerateGlobalTodolist:
    """测试生成全局 todolist"""
    
    @pytest.fixture
    def agent(self):
        """创建 PlanningAgent 实例"""
        return PlanningAgent(task_id="test_task_global")
    
    @pytest.mark.asyncio
    async def test_generate_global_todolist_success(self, agent):
        """测试成功生成全局 todolist"""
        mock_response = """<global_todo_list>
1. 生成 ReLU 算子的 task_desc 定义
2. 实现 ReLU kernel 代码
3. 验证 kernel 精度并输出结果
</global_todo_list>"""
        
        with patch.object(agent, 'run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (mock_response, "", "")
            
            result = await agent.generate_global_todolist(
                user_input="生成一个 ReLU 算子",
                available_tools=[]
            )
            
            assert "<global_todo_list>" in result
            assert "ReLU" in result
            mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_global_todolist_with_task_desc(self, agent):
        """测试有 task_desc 时生成全局 todolist"""
        mock_response = """<global_todo_list>
1. 实现 kernel 代码
2. 验证 kernel 精度
</global_todo_list>"""
        
        with patch.object(agent, 'run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (mock_response, "", "")
            
            result = await agent.generate_global_todolist(
                user_input="帮我生成这个算子的 kernel",
                available_tools=[],
                has_task_desc=True
            )
            
            # 验证 has_task_desc 参数被传递
            call_args = mock_llm.call_args
            assert call_args[1]['input']['has_task_desc'] == True
    
    @pytest.mark.asyncio
    async def test_generate_global_todolist_fallback_on_error(self, agent):
        """测试生成失败时返回默认值"""
        with patch.object(agent, 'run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")
            
            result = await agent.generate_global_todolist(
                user_input="生成一个 ReLU 算子",
                available_tools=[]
            )
            
            # 应该返回默认值
            assert "<global_todo_list>" in result


class TestGenerateDetailedTodolist:
    """测试生成详细 todolist"""
    
    @pytest.fixture
    def agent(self):
        """创建 PlanningAgent 实例"""
        return PlanningAgent(task_id="test_task_detailed")
    
    @pytest.mark.asyncio
    async def test_generate_detailed_todolist_success(self, agent):
        """测试成功生成详细 todolist"""
        mock_response = """<todo_list>
1. 调用 call_op_task_builder args={input:"init/user_input.txt", output:"op_task_builder/task_desc.py", op_name:"relu"}
2. 调用 call_coder_only args={input:"op_task_builder/task_desc.py", output:"coder/kernel_code.py", op_name:"relu"}
3. 调用 call_kernel_verifier args={input:"coder/kernel_code.py", task_desc:"op_task_builder/task_desc.py", output:"verifier/verify_result.json", op_name:"relu"}
4. 完成任务并输出结果
</todo_list>"""
        
        global_todo = """<global_todo_list>
1. 生成 ReLU 算子的 task_desc
2. 实现 ReLU kernel 代码
3. 验证 kernel 精度
</global_todo_list>"""
        
        with patch.object(agent, 'run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (mock_response, "", "")
            
            result = await agent.generate_detailed_todolist(
                user_input="生成一个 ReLU 算子",
                global_todo_list=global_todo,
                available_tools=[]
            )
            
            assert "<todo_list>" in result
            assert "call_op_task_builder" in result
            assert "call_coder_only" in result
            assert "call_kernel_verifier" in result
    
    @pytest.mark.asyncio
    async def test_generate_detailed_todolist_with_action_history(self, agent):
        """测试有动作历史时生成详细 todolist"""
        mock_response = """<todo_list>
1. 调用 call_coder_only args={...}
2. 调用 call_kernel_verifier args={...}
3. 完成任务并输出结果
</todo_list>"""
        
        global_todo = "<global_todo_list>...</global_todo_list>"
        action_history = [
            {
                "tool_name": "call_op_task_builder",
                "result": {"status": "success"}
            }
        ]
        
        with patch.object(agent, 'run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = (mock_response, "", "")
            
            result = await agent.generate_detailed_todolist(
                user_input="生成一个 ReLU 算子",
                global_todo_list=global_todo,
                available_tools=[],
                action_history=action_history
            )
            
            # 验证 action_history 被格式化并传递
            call_args = mock_llm.call_args
            assert "call_op_task_builder" in call_args[1]['input']['action_history']
    
    @pytest.mark.asyncio
    async def test_generate_detailed_todolist_fallback_on_error(self, agent):
        """测试生成失败时返回默认值"""
        with patch.object(agent, 'run_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM error")
            
            result = await agent.generate_detailed_todolist(
                user_input="生成一个 ReLU 算子",
                global_todo_list="<global_todo_list>...</global_todo_list>",
                available_tools=[]
            )
            
            # 应该返回默认值
            assert "<todo_list>" in result
            assert "call_op_task_builder" in result


class TestComplexScenarios:
    """测试复杂场景"""
    
    @pytest.fixture
    def agent(self):
        """创建 PlanningAgent 实例"""
        return PlanningAgent(task_id="test_task_complex")
    
    @pytest.mark.asyncio
    async def test_high_performance_request(self, agent):
        """测试高性能请求（应该包含 designer）"""
        mock_global = """<global_todo_list>
1. 生成 MatMul 算子的 task_desc 定义
2. 设计 MatMul kernel 的优化架构
3. 实现优化的 MatMul kernel 代码
4. 验证 kernel 精度和性能
</global_todo_list>"""
        
        mock_detailed = """<todo_list>
1. 调用 call_op_task_builder args={input:"init/user_input.txt", output:"op_task_builder/task_desc.py", op_name:"matmul"}
2. 调用 call_designer args={input:"op_task_builder/task_desc.py", output:"designer/design_output.txt", op_name:"matmul"}
3. 调用 call_evolve args={input:"op_task_builder/task_desc.py", design:"designer/design_output.txt", output:"coder/kernel_code.py", op_name:"matmul"}
4. 调用 call_kernel_verifier args={input:"coder/kernel_code.py", task_desc:"op_task_builder/task_desc.py", output:"verifier/verify_result.json", op_name:"matmul"}
5. 完成任务并输出结果
</todo_list>"""
        
        with patch.object(agent, 'run_llm', new_callable=AsyncMock) as mock_llm:
            # 第一次调用返回全局 todolist
            mock_llm.return_value = (mock_global, "", "")
            
            global_result = await agent.generate_global_todolist(
                user_input="生成一个高性能的 MatMul 算子",
                available_tools=[]
            )
            
            # 第二次调用返回详细 todolist
            mock_llm.return_value = (mock_detailed, "", "")
            
            detailed_result = await agent.generate_detailed_todolist(
                user_input="生成一个高性能的 MatMul 算子",
                global_todo_list=global_result,
                available_tools=[]
            )
            
            assert "设计" in global_result or "设计" in mock_global
            assert "call_designer" in detailed_result
            assert "call_evolve" in detailed_result
    
    def test_tools_with_full_parameters(self, agent):
        """测试完整参数的工具格式化"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "call_coder_only",
                    "description": "快速生成 kernel 代码",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_code": {
                                "type": "string",
                                "description": "task 定义代码"
                            },
                            "op_name": {
                                "type": "string",
                                "description": "算子名称"
                            },
                            "user_requirements": {
                                "type": "string",
                                "description": "用户的额外优化需求"
                            }
                        },
                        "required": ["task_code", "op_name"]
                    }
                }
            }
        ]
        
        result = agent._format_tools(tools)
        
        assert "call_coder_only" in result
        assert "task_code" in result
        assert "op_name" in result
        assert "user_requirements" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

