from akg_agents.op.config.config_validator import load_config
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.environment_check import check_env_for_task
import asyncio
import os
from pathlib import Path


def get_op_name():
    return '069_rms_norm'


def get_task_desc(case_dir):
    with open(case_dir / "definition.json", "r", encoding="utf-8") as f:
        def_json = f.read()
    with open(case_dir / "reference.py", "r", encoding="utf-8") as f:
        ref_py = f.read()

    workload_sample = ""
    workload_file = case_dir / "workload.jsonl"
    if workload_file.exists():
        import json
        with open(workload_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            first = json.loads(lines[0])
            workload_sample = (
                f"\n\n## workload 示例（共 {len(lines)} 组，以下为第 1 组）\n"
                f"```json\n{json.dumps(first, indent=2)}\n```"
            )

    return (
        f"请实现一个 Triton CUDA 算子。\n\n"
        f"## definition.json\n```json\n{def_json}\n```\n\n"
        f"## reference.py\n```python\n{ref_py}\n```"
        f"{workload_sample}\n\n"
        f"注意：请使用 Triton 编写 kernel，并将其封装在 ModelNew 类的 forward 方法中。"
    )


async def run_sol_triton_single():
    op_name = get_op_name()
    
    case_dir = Path(__file__).parent.parent.parent.parent / "thirdparty" / "sol-execbench" / "data" / "benchmark" / "L1" / op_name
    task_desc = get_task_desc(case_dir)

    task_pool = TaskPool()
    
    await register_local_worker([0], backend="cuda", arch="rtx3090")
    
    config = load_config("triton_cuda", backend="cuda")
    config["sol_problem_dir"] = str(case_dir)
    
    check_env_for_task("torch", "cuda", "triton_cuda", config)

    task = LangGraphTask(
        op_name=op_name,
        task_desc=task_desc,
        task_id="sol_single_069",
        dsl="triton_cuda",
        backend="cuda",
        arch="rtx3090",
        config=config,
        framework="torch",
        workflow="coder_only_workflow",
        bench_type="sol"
    )

    task_pool.create_task(task.run)
    results = await task_pool.wait_all()
    for op_name, result, _ in results:
        if result:
            print(f"Task {op_name} passed")
        else:
            print(f"Task {op_name} failed")

if __name__ == "__main__":
    asyncio.run(run_sol_triton_single())
