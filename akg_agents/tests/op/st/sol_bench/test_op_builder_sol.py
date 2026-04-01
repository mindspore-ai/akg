import pytest
import os
import json
from akg_agents.op.workflows.op_task_builder_workflow import run_op_task_builder
from akg_agents.op.langgraph_op.op_task_builder_state import OpTaskBuilderStatus
from akg_agents.op.config.config_validator import load_config

def get_test_config() -> dict:
    try:
        config = load_config(config_path="./python/akg_agents/op/config/triton_ascend_coderonly_config.yaml")
        agent_model_config = config.get("agent_model_config", {})
        if "op_task_builder" not in agent_model_config:
            agent_model_config["op_task_builder"] = agent_model_config.get("coder", "standard")
            config["agent_model_config"] = agent_model_config
        return config
    except Exception:
        return {
            "agent_model_config": {
                "op_task_builder": "standard",
            },
            "log_dir": os.path.expanduser("~/akg_agents_logs/test_op_builder_sol"),
            "op_task_builder_max_iterations": 5
        }

@pytest.mark.asyncio
async def test_op_builder_sol_format():
    """测试 OpTaskBuilder 生成 SOL-ExecBench 格式的任务"""
    
    config = get_test_config()
    
    # 运行 OpTaskBuilder 工作流
    result = await run_op_task_builder(
        user_input="实现一个简单的 ReLU 算子，输入大小为 [16, 16384]，数据类型为 float32，不需要其他参数。",
        config=config,
        framework="torch",
        backend="cpu",
        arch="x86_64",
        dsl="cpp",
        bench_type="sol"
    )
    
    # 验证结果状态
    print(f"Status: {result.get('status')}")
    if result.get('status') == OpTaskBuilderStatus.NEED_CLARIFICATION:
        print(f"Clarification question: {result.get('agent_message')}")
    
    assert result.get("status") in [
        OpTaskBuilderStatus.READY,
        OpTaskBuilderStatus.NEED_CLARIFICATION,
    ], f"意外的状态: {result.get('status')}"
    
    if result.get("status") == OpTaskBuilderStatus.READY:
        generated_code = result.get("generated_task_desc", "")
        assert generated_code, "未生成任务代码"
        
        # 由于 SOL 格式的输出是一个包含 definition.json, workload.jsonl, reference.py 的 JSON 字符串
        try:
            sol_data = json.loads(generated_code)
            assert "definition.json" in sol_data, "缺少 definition.json"
            assert "workload.jsonl" in sol_data, "缺少 workload.jsonl"
            assert "reference.py" in sol_data, "缺少 reference.py"
            
            # 进一步检查关键要素是否齐全
            def_json = sol_data["definition.json"]
            if isinstance(def_json, str):
                def_json = json.loads(def_json)
            assert "inputs" in def_json, "definition.json 缺少 inputs 字段"
            assert "outputs" in def_json, "definition.json 缺少 outputs 字段"
            
            ref_py = sol_data["reference.py"]
            assert "def run" in ref_py, "reference.py 缺少 run 函数"
            
        except json.JSONDecodeError:
            pytest.fail(f"生成的代码不是有效的 JSON 格式: {generated_code}")