# Copyright 2025 Huawei Technologies Co., Ltd
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

import asyncio
import argparse
import json
from akg_agents.op.evolve import evolve
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task
from akg_agents.op.utils.evolve.runner_manager import RunnerConfig, print_evolve_config, print_evolution_result
from akg_agents import get_project_root
from pathlib import Path
from akg_agents.core.worker.manager import register_local_worker


DEFAULT_OP_NAME = 'akg_agents_relu'

DEFAULT_TASK_DESC = '''
import torch
import torch.nn as nn
import torch_npu

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.rand(batch_size, dim, device='npu')
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
'''


def get_sol_task_desc(case_dir: Path) -> str:
    """从 SOL 数据集目录构建 task_desc"""
    with open(case_dir / "definition.json", "r", encoding="utf-8") as f:
        def_json = f.read()
    with open(case_dir / "reference.py", "r", encoding="utf-8") as f:
        ref_py = f.read()

    workload_sample = ""
    workload_file = case_dir / "workload.jsonl"
    if workload_file.exists():
        with open(workload_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        if lines:
            first = json.loads(lines[0])
            workload_sample = (
                f"\n\n## workload 示例（共 {len(lines)} 组，以下为第 1 组）\n"
                f"```json\n{json.dumps(first, indent=2)}\n```"
            )

    return (
        f"请实现一个 Triton Ascend 算子。\n\n"
        f"## definition.json\n```json\n{def_json}\n```\n\n"
        f"## reference.py\n```python\n{ref_py}\n```"
        f"{workload_sample}\n\n"
        f"注意：请使用 Triton 编写 kernel，并将其封装在 ModelNew 类的 forward 方法中。"
    )


def get_sol_op_name(case_dir: Path) -> str:
    """从 SOL definition.json 提取算子名称"""
    def_file = case_dir / "definition.json"
    if def_file.exists():
        with open(def_file, "r", encoding="utf-8") as f:
            definition = json.load(f)
        return definition.get("name", case_dir.name)
    return case_dir.name


async def run_torch_evolve_triton_ascend(args):
    """运行Triton Ascend进化示例 (使用 WorkerManager 新架构)"""

    # 创建配置对象并设置硬编码参数
    config = RunnerConfig()

    # 基础配置
    config.dsl = "triton_ascend"
    config.framework = "torch"
    config.backend = "ascend"
    config.arch = "ascend910b4"

    # 进化参数
    config.max_rounds = 3
    config.parallel_num = 4

    # 岛屿模型参数
    config.num_islands = 2
    config.migration_interval = 2
    config.elite_size = 5
    config.parent_selection_prob = 0.5

    # 设备配置
    config.device_list = [0, 1, 2, 3]

    # 配置文件路径
    config.config_path = str(Path(get_project_root()) / "op" / "config" / "triton_ascend_evolve_config.yaml")

    # 根据输入模式确定 op_name / task_desc / bench_type
    if args.sol_problem_dir:
        case_dir = Path(args.sol_problem_dir).resolve()
        if not (case_dir / "definition.json").exists():
            raise FileNotFoundError(f"SOL 数据集目录缺少 definition.json: {case_dir}")
        config.op_name = get_sol_op_name(case_dir)
        config.task_desc = get_sol_task_desc(case_dir)
        bench_type = "sol"
        print(f"SOL 模式: 数据集目录={case_dir}")
    else:
        config.op_name = DEFAULT_OP_NAME
        config.task_desc = DEFAULT_TASK_DESC
        bench_type = "kernelbench"

    # 打印配置信息
    print_evolve_config(config.op_name, config)

    # 初始化资源池
    task_pool = TaskPool(max_concurrency=config.parallel_num)
    
    await register_local_worker(config.device_list, backend=config.backend, arch=config.arch)

    try:
        loaded_config = load_config(config_path=config.config_path)
    except ValueError:
        print(f"Config file {config.config_path} not found, using default.")
        loaded_config = load_config(dsl=config.dsl, backend=config.backend)

    # SOL 模式：注入 bench_type 和 sol_problem_dir 到 config
    if bench_type == "sol":
        loaded_config["bench_type"] = "sol"
        loaded_config["sol_problem_dir"] = str(case_dir)

    check_env_for_task(config.framework, config.backend, config.dsl, loaded_config)

    print("开始进化过程...")
    evolution_result = await evolve(
        op_name=config.op_name,
        task_desc=config.task_desc,
        dsl=config.dsl,
        framework=config.framework,
        backend=config.backend,
        arch=config.arch,
        config=loaded_config,
        task_pool=task_pool,
        max_rounds=config.max_rounds,
        parallel_num=config.parallel_num,
        num_islands=config.num_islands,
        migration_interval=config.migration_interval,
        elite_size=config.elite_size,
        parent_selection_prob=config.parent_selection_prob
    )

    # 打印结果
    print_evolution_result(evolution_result, config)


def main():
    parser = argparse.ArgumentParser(
        description="运行进化搜索示例 (Triton Ascend)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认配置（KernelBench 模式）
  python run_torch_evolve_triton_ascend.py

  # SOL-ExecBench 模式
  python run_torch_evolve_triton_ascend.py --sol-problem-dir examples/kernel_related/mock_sol_relu
        """
    )
    parser.add_argument(
        "--sol-problem-dir",
        type=str,
        default=None,
        help="SOL-ExecBench 数据集目录路径（包含 definition.json、reference.py、workload.jsonl）。"
             "指定后自动切换为 SOL bench_type。"
    )
    args = parser.parse_args()
    asyncio.run(run_torch_evolve_triton_ascend(args))


if __name__ == "__main__":
    main()

