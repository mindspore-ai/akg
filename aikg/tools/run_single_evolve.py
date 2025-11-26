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

"""
单任务进化执行脚本

用法:
  python run_single_evolve.py                                    # 使用默认配置
  python run_single_evolve.py <config_file>                      # 使用YAML配置文件
  python run_single_evolve.py <op_name> <task_file> [config_file]  # 批量runner模式
"""

import sys
import asyncio
import os
from pathlib import Path

# 添加项目根目录到sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_kernel_generator import get_project_root
from ai_kernel_generator.utils.evolve.runner_manager import (
    RunnerConfig,
    apply_custom_task_config,
    load_task_description,
    run_single_evolve,
    print_evolve_config
)
from ai_kernel_generator.core.worker.manager import register_worker


def print_usage():
    """打印使用说明"""
    print("用法:")
    print("  python run_single_evolve.py                                                    # 使用默认配置")
    print("  python run_single_evolve.py <config_file>                                      # 使用YAML配置文件")
    print("  python run_single_evolve.py <op_name> <task_file> <device> [config_file]      # batch runner简化模式")


def parse_default_config():
    """解析默认配置"""
    project_root = get_project_root()
    config_path = os.path.join(project_root, "config", "evolve_config.yaml")

    try:
        config = RunnerConfig.from_yaml(config_path)
        op_name = config.op_name
        task_desc = config.task_desc

        print(f"使用默认配置文件: {config_path}")
        print(f"算子名称: {op_name}")
        print(f"任务描述文件: {task_desc}")

        return op_name, task_desc, config
    except Exception as e:
        print(f"无法加载默认配置文件 {config_path}: {e}")
        print("使用内置默认配置")
        config = RunnerConfig()
        op_name = config.op_name
        task_desc = config.task_desc

        return op_name, task_desc, config


def parse_config_file_mode(config_path: str):
    """解析配置文件模式"""
    try:
        config = RunnerConfig.from_yaml(config_path)
        op_name = config.op_name
        task_desc = config.task_desc

        print(f"使用配置文件: {config_path}")
        print(f"算子名称: {op_name}")
        print(f"任务描述文件: {task_desc}")

        return op_name, task_desc, config
    except Exception as e:
        print(f"无法加载配置文件 {config_path}: {e}")
        sys.exit(1)


def parse_batch_runner_mode(args):
    """解析批量运行器模式"""
    op_name = args[1]
    task_file = args[2]
    device = int(args[3])
    project_root = get_project_root()
    config_path = os.path.join(project_root, "config", "evolve_config.yaml")
    
    config = RunnerConfig()

    # 如果提供了配置文件路径
    if len(args) == 5:
        config_path = args[4]

        try:
            file_config = RunnerConfig.from_yaml(config_path, skip_task_config=True)
            for key, value in file_config.to_dict().items():
                setattr(config, key, value)

            apply_custom_task_config(config, config_path, op_name)

        except Exception as e:
            print(f"警告: 无法加载配置文件 {config_path}: {e}")
        
    try:
        file_config = RunnerConfig.from_yaml(config_path, skip_task_config=True)
        for key, value in file_config.to_dict().items():
            setattr(config, key, value)
        
        apply_custom_task_config(config, config_path, op_name)
            
    except Exception as e:
        print(f"警告: 无法加载配置文件 {config_path}: {e}")

    # 设置设备 (仅用于注册 Worker)
    config.device_list = [device]

    # 读取任务描述文件
    task_desc = load_task_description(task_file)

    print(f"任务: {op_name}")
    print(f"任务文件: {task_file}")
    print(f"设备: {config.device_list}")
    print(f"配置: {config.max_rounds}轮/{config.parallel_num}并行")
    print(f"基础参数: {config.dsl}/{config.framework}/{config.backend}/{config.arch}")

    # 岛屿模型配置
    if config.num_islands > 1:
        print(f"岛屿数量: {config.num_islands}")
        if config.migration_interval > 0:
            print(f"迁移间隔: {config.migration_interval}")
        if config.elite_size > 0:
            print(f"精英数量: {config.elite_size}")
            print(f"父代选择概率: {config.parent_selection_prob}")

    return op_name, task_desc, config


async def run_wrapper(op_name, task_desc, config):
    """包装运行函数，负责注册Worker"""
    # 注册 Worker
    await register_worker(
        backend=config.backend,
        arch=config.arch,
        device_ids=config.device_list
    )
    
    return await run_single_evolve(op_name=op_name, task_desc=task_desc, evolve_config=config)


def main():
    """主函数"""
    # 解析命令行参数
    if len(sys.argv) == 1:
        # 无参数模式：使用默认配置文件
        op_name, task_desc, config = parse_default_config()

    elif len(sys.argv) == 2:
        # 配置文件模式：从YAML配置文件加载
        config_path = sys.argv[1]
        op_name, task_desc, config = parse_config_file_mode(config_path)

    elif len(sys.argv) >= 4:
        # batch_runner简化模式: op_name task_file device [config_file]
        op_name, task_desc, config = parse_batch_runner_mode(sys.argv)

    else:
        print_usage()
        sys.exit(1)

    # 运行任务
    try:
        result = asyncio.run(run_wrapper(op_name=op_name, task_desc=task_desc, config=config))

        if result:
            print("\n进化式算子生成成功完成!")
            successful_tasks = result.get('successful_tasks', 0)
            if successful_tasks > 0:
                print(f"成功生成了 {successful_tasks} 个有效的算子实现")
            else:
                print("未能生成成功的算子实现，请检查配置和任务描述")
        else:
            print("\n进化过程失败，请检查日志获取详细信息")

    except Exception as e:
        print(f"Error occurred during evolution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

