import os
import json
import pytest
from pathlib import Path
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.langgraph_op.task import LangGraphTask as AIKGTask
from akg_agents.core.worker.manager import register_local_worker
from ..utils import get_device_id
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task

device_id = get_device_id()

@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.cpp
@pytest.mark.cpu
@pytest.mark.x86_64
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_bench_sol_cpu_cpp():
    """测试 SOL-ExecBench 格式 - PyTorch C++ CPU (端到端)"""
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "x86_64"
    
    config = load_config("cpp", backend="cpu")
    
    # 获取 _sol_relu 的路径
    current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sol_problem_dir = os.path.join(current_dir, "examples", "kernel_related", "mock_sol_relu")
    
    config["sol_problem_dir"] = sol_problem_dir
    config["verify_timeout"] = 300
    
    check_env_for_task(framework, backend, dsl, config)
    
    # 注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)
    
    # 读取 definition.json、reference.py、workload.jsonl 作为 task_desc
    with open(os.path.join(sol_problem_dir, "definition.json"), "r", encoding="utf-8") as f:
        def_json = f.read()
    with open(os.path.join(sol_problem_dir, "reference.py"), "r", encoding="utf-8") as f:
        ref_py = f.read()

    workload_sample = ""
    workload_path = os.path.join(sol_problem_dir, "workload.jsonl")
    if os.path.exists(workload_path):
        with open(workload_path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            first = json.loads(lines[0])
            workload_sample = (
                f"\n\n## workload 示例（共 {len(lines)} 组，以下为第 1 组）\n"
                f"```json\n{json.dumps(first, indent=2)}\n```"
            )

    task_desc = (
        f"请实现一个 C++ CPU 算子。\n\n"
        f"## definition.json\n```json\n{def_json}\n```\n\n"
        f"## reference.py\n```python\n{ref_py}\n```"
        f"{workload_sample}\n\n"
        f"注意：请使用 torch.utils.cpp_extension.load_inline 编译 C++ 代码，并将其封装在 ModelNew 类中。"
    )
    
    task = AIKGTask(
        op_name="_relu",
        task_desc=task_desc,
        task_id="test_bench_sol_cpu_001",
        backend=backend,
        arch=arch,
        dsl=dsl,
        config=config,
        framework=framework,
        workflow="coder_only_workflow",
        bench_type="sol"
    )
    
    result = await task.run()
    
    # task.run() 返回 (op_name, success, state_dict)
    assert result[1] is True, f"端到端执行失败: {result[2].get('error_message')}"
