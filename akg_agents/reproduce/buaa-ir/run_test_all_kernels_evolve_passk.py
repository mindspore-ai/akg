import asyncio
import os
import re
import argparse
from pathlib import Path

from akg_agents.op.evolve import evolve
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.core.worker.manager import register_worker
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task
from akg_agents.op.utils.evolve.runner_manager import RunnerConfig, print_evolve_config, print_evolution_result
from akg_agents import get_project_root

def get_sorted_py_files(folder_path: str):
    """获取按数字前缀排序的 Python 文件列表"""
    py_files = []
    for file in os.listdir(folder_path):
        if file.endswith(".py"):
            match = re.match(r"(\d+)_.*\.py", file)
            if match:
                index = int(match.group(1))
                py_files.append((index, file))
    py_files.sort(key=lambda x: x[0])
    return [os.path.join(folder_path, f[1]) for f in py_files]


def extract_op_name_from_filename(filename: str):
    """提取 op_name，去掉前缀数字部分，并在前面加一个下划线"""
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    # name = re.sub(r"^\d+_", "_", name)
    name = '_' + name
    return name


def read_task_desc(filepath: str):
    """读取文件内容作为 task_desc"""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


async def run_torch_evolve_triton(args):
    """从指定文件夹读取任务并依次运行 Triton 进化"""

    py_files = get_sorted_py_files(args.folder)
    if not py_files:
        print(f"❌ 未在 {args.folder} 找到符合规则的 Python 文件")
        return

    # 如果指定 start/end，只取指定范围
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index - 1 if args.start_index else 0
        end = args.end_index if args.end_index else len(py_files)
        py_files = py_files[start:end]

    flag = 0

    for file_path in py_files:
        op_name = extract_op_name_from_filename(file_path)
        task_desc = read_task_desc(file_path)

        print(f"\n🚀 正在处理任务: {op_name}")
        print(f"📄 来源文件: {file_path}")

        # 创建配置对象
        config = RunnerConfig()
        config.dsl = args.dsl
        config.framework = args.framework
        config.backend = args.backend
        config.arch = args.arch

        config.max_rounds = args.max_rounds
        config.parallel_num = args.parallel_num
        config.num_islands = args.num_islands
        config.migration_interval = args.migration_interval
        config.elite_size = args.elite_size
        config.parent_selection_prob = args.parent_selection_prob
        config.device_list = [args.device]
        config.config_path = str(Path(get_project_root()) / "op" / "config" / "triton_cuda_evolve_config.yaml")

        config.op_name = op_name
        config.task_desc = task_desc

        print_evolve_config(config.op_name, config)

        # 初始化资源池
        task_pool = TaskPool(max_concurrency=config.parallel_num)
        
        # device_pool = DevicePool(config.device_list)
        print(f"🔗 注册 LocalWorker: devices={config.device_list}")
        await register_worker(
            backend=config.backend,
            arch=config.arch,
            device_ids=config.device_list
        )
        print()

        if flag == 0:
            config2 = load_config(config_path=config.config_path)
            flag = 1
        # 环境检查
        is_remote = False
        check_env_for_task(config.framework, config.backend, config.dsl, config2, is_remote=is_remote)

        # 执行进化任务
        print(f"开始进化 {op_name} ...")

        evolution_result = await evolve(
            op_name=config.op_name,
            task_desc=config.task_desc,
            dsl=config.dsl,
            framework=config.framework,
            backend=config.backend,
            arch=config.arch,
            # config=load_config(config_path=config.config_path),
            config=config2,
            # device_pool=device_pool,
            task_pool=task_pool,
            max_rounds=config.max_rounds,
            parallel_num=config.parallel_num,
            num_islands=config.num_islands,
            migration_interval=config.migration_interval,
            elite_size=config.elite_size,
            parent_selection_prob=config.parent_selection_prob,
        )

        print_evolution_result(evolution_result, config)


def parse_args():
    parser = argparse.ArgumentParser(description="Run Triton evolve tasks from a folder.")
    parser.add_argument("--folder", type=str, required=True, help="任务文件所在文件夹路径")
    parser.add_argument("--max_rounds", type=int, default=2, help="最大进化轮次")
    parser.add_argument("--parallel_num", type=int, default=1, help="并行任务数量")
    parser.add_argument("--num_islands", type=int, default=1, help="岛屿数量")
    parser.add_argument("--migration_interval", type=int, default=2, help="迁移间隔轮数")
    parser.add_argument("--elite_size", type=int, default=5, help="精英个体数量")
    parser.add_argument("--parent_selection_prob", type=float, default=0.5, help="父代选择概率")
    parser.add_argument("--device", type=int, default=0, help="CUDA 设备编号")
    parser.add_argument("--arch", type=str, default="a100", help="GPU 架构 (如 a100, h100)")
    parser.add_argument("--backend", type=str, default="cuda", help="后端类型")
    parser.add_argument("--framework", type=str, default="torch", help="深度学习框架")
    parser.add_argument("--dsl", type=str, default="triton_cuda", help="DSL 类型")
    parser.add_argument("--start_index", type=int, help="从第 N 个任务开始（可选）")
    parser.add_argument("--end_index", type=int, help="运行到第 N 个任务结束（可选）")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_torch_evolve_triton(args))
