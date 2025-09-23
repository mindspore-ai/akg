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
import sys
import os
from pathlib import Path
import json
import yaml
from datetime import datetime
from ai_kernel_generator.core.evolve import evolve
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task
from ai_kernel_generator import get_project_root
from typing import Optional, List, Dict, Any


# ============================================================================ #
# 配置参数类
# ============================================================================ #


class EvolveConfig:
    """进化配置参数类"""

    def __init__(self):
        # 基本参数
        self.dsl = "triton"
        self.framework = "torch"
        self.backend = "cuda"
        self.arch = "a100"

        # 进化参数
        self.max_rounds = 5
        self.parallel_num = 4

        # 岛屿模型参数
        self.num_islands = 1  # 设置为1或更小值可禁用岛屿模型
        self.migration_interval = 0  # 设置为0可禁用迁移
        self.elite_size = 0  # 设置为0可禁用精英机制
        self.parent_selection_prob = 0.5  # 父代选择概率

        # 设备配置
        self.device_list = [0]

        # 配置文件路径
        self.config_path = str(Path(get_project_root()) / "config" / "vllm_triton_evolve_config.yaml")

        # 任务配置
        self.op_name = "relu_op"
        self.task_desc = "Path/to/your/tasks/relu_task.py"

    @classmethod
    def from_yaml(cls, config_path: str, skip_task_config: bool = False):
        """从YAML配置文件加载配置
        
        Args:
            config_path: 配置文件路径
            skip_task_config: 是否跳过任务配置（用于批量调用模式）
        """
        config = cls()
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)

            # 基础配置
            if 'base' in yaml_config:
                base = yaml_config['base']
                config.dsl = base.get('dsl', config.dsl)
                config.framework = base.get('framework', config.framework)
                config.backend = base.get('backend', config.backend)
                config.arch = base.get('arch', config.arch)
            else:
                raise ValueError("base section not found in config file")

            # 进化参数
            if 'evolve' in yaml_config:
                evolve_config = yaml_config['evolve']
                config.max_rounds = evolve_config.get('max_rounds', config.max_rounds)
                config.parallel_num = evolve_config.get('parallel_num', config.parallel_num)
            else:
                raise ValueError("evolve section not found in config file")

            # 岛屿模型配置
            if 'island' in yaml_config:
                island_config = yaml_config['island']
                config.num_islands = island_config.get('num_islands', config.num_islands)
                config.migration_interval = island_config.get('migration_interval', config.migration_interval)
                config.elite_size = island_config.get('elite_size', config.elite_size)
                config.parent_selection_prob = island_config.get('parent_selection_prob', config.parent_selection_prob)
            else:
                raise ValueError("island section not found in config file")

            # 设备配置
            if 'devices' in yaml_config:
                config.device_list = yaml_config['devices'].get('device_list', config.device_list)
            else:
                raise ValueError("devices section not found in config file")

            # 任务配置（仅在非批量调用模式下加载）
            if not skip_task_config and 'task' in yaml_config:
                task_config = yaml_config['task']
                config.op_name = task_config.get('op_name', config.op_name)

                # 处理task_desc，从文件路径读取
                task_desc_value = task_config.get('task_desc', config.task_desc)
                if task_desc_value and isinstance(task_desc_value, str):
                    try:
                        # 作为文件路径读取
                        with open(task_desc_value, 'r', encoding='utf-8') as f:
                            config.task_desc = f.read().strip()
                    except Exception as e:
                        print(f"Error: Failed to read task description file {task_desc_value}: {e}")
                        config.task_desc = None

            return config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}, using default config: {e}")
            return config

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'dsl': self.dsl,
            'framework': self.framework,
            'backend': self.backend,
            'arch': self.arch,
            'max_rounds': self.max_rounds,
            'parallel_num': self.parallel_num,
            'num_islands': self.num_islands,
            'migration_interval': self.migration_interval,
            'elite_size': self.elite_size,
            'parent_selection_prob': self.parent_selection_prob,
            'device_list': self.device_list,
            'config_path': self.config_path,
            'op_name': self.op_name,
            'task_desc': self.task_desc
        }


async def run_custom_evolve(op_name: str = None, task_desc: str = None, evolve_config: EvolveConfig = None):
    """运行自定义任务的进化过程

    Args:
        op_name: 算子名称，如果为None则使用evolve_config中的配置
        task_desc: 任务描述，如果为None则使用evolve_config中的配置
        evolve_config: 进化配置类实例，如果为None则创建默认配置
    """
    # 使用传入的配置或创建默认配置
    if evolve_config is None:
        evolve_config = EvolveConfig()

    # 如果op_name或task_desc为None，尝试从配置中获取
    if op_name is None:
        op_name = evolve_config.op_name
    if task_desc is None:
        task_desc = evolve_config.task_desc

    print("="*80)
    print("AI KERNEL GENERATOR - 统一进化式算子生成")
    print("="*80)
    print(f"算子名称: {op_name}")
    print(f"实现类型: {evolve_config.dsl}")
    print(f"框架: {evolve_config.framework}")
    print(f"后端: {evolve_config.backend}")
    print(f"架构: {evolve_config.arch}")
    print(f"进化轮数: {evolve_config.max_rounds}")
    print(f"并行任务数: {evolve_config.parallel_num}")

    # 岛屿模型配置说明
    if evolve_config.num_islands <= 1:
        print("岛屿模型: 禁用（简单进化模式）")
    else:
        print(f"岛屿数量: {evolve_config.num_islands}")
        if evolve_config.migration_interval <= 0:
            print("迁移: 禁用")
        else:
            print(f"迁移间隔: {evolve_config.migration_interval}")

    if evolve_config.elite_size <= 0:
        print("精英机制: 禁用")
    else:
        print(f"精英数量: {evolve_config.elite_size}")

    if evolve_config.num_islands > 1 and evolve_config.elite_size > 0:
        print(f"父代选择概率: {evolve_config.parent_selection_prob}")
    print("="*80)

    # 初始化资源
    task_pool = TaskPool(max_concurrency=evolve_config.parallel_num)
    device_pool = DevicePool(evolve_config.device_list)

    config = load_config(config_path=evolve_config.config_path)
    check_env_for_task(evolve_config.framework, evolve_config.backend, evolve_config.dsl, config)

    # 运行进化过程
    print("开始进化过程...")
    evolution_result = await evolve(
        op_name=op_name,
        task_desc=task_desc,
        dsl=evolve_config.dsl,
        framework=evolve_config.framework,
        backend=evolve_config.backend,
        arch=evolve_config.arch,
        config=config,
        device_pool=device_pool,
        task_pool=task_pool,
        max_rounds=evolve_config.max_rounds,
        parallel_num=evolve_config.parallel_num,
        num_islands=evolve_config.num_islands,
        migration_interval=evolve_config.migration_interval,
        elite_size=evolve_config.elite_size,
        parent_selection_prob=evolve_config.parent_selection_prob
    )

    # 检查进化结果是否有效
    if not evolution_result:
        print("\n❌ 进化过程返回空结果")
        return None

    # 输出进化结果
    print("\n" + "="*80)
    print("进化完成！最终结果汇总:")
    print("="*80)
    print(f"算子名称: {evolution_result.get('op_name', 'Unknown')}")
    print(f"总轮数: {evolution_result.get('total_rounds', 0)}")
    print(f"总任务数: {evolution_result.get('total_tasks', 0)}")
    print(f"成功任务数: {evolution_result.get('successful_tasks', 0)}")
    print(f"最终成功率: {evolution_result.get('final_success_rate', 0.0):.2%}")
    print(f"最佳成功率: {evolution_result.get('best_success_rate', 0.0):.2%}")
    print(f"实现类型: {evolution_result.get('implementation_type', 'Unknown')}")
    print(f"框架: {evolution_result.get('framework', 'Unknown')}")
    print(f"后端: {evolution_result.get('backend', 'Unknown')}")
    print(f"架构: {evolution_result.get('architecture', 'Unknown')}")

    # 岛屿信息
    island_info = evolution_result.get('island_info', {})
    if island_info:
        num_islands_used = island_info.get('num_islands', 'N/A')
        if num_islands_used <= 1:
            print("进化模式: 简单进化（无岛屿模型）")
        else:
            print(f"岛屿数量: {num_islands_used}")
            migration_interval_used = island_info.get('migration_interval', 'N/A')
            if migration_interval_used <= 0:
                print("迁移: 禁用")
            else:
                print(f"迁移间隔: {migration_interval_used}")

            elite_size_used = island_info.get('elite_size', 'N/A')
            if elite_size_used <= 0:
                print("精英机制: 禁用")
            else:
                print(f"精英数量: {elite_size_used}")

    # 显示存储目录信息
    storage_dir = evolution_result.get('storage_dir', '')
    if storage_dir:
        print(f"存储目录: {storage_dir}")

    # 显示最佳实现
    best_implementations = evolution_result.get('best_implementations', [])
    if best_implementations:
        print(f"\n最佳实现 (前{len(best_implementations)}个):")
        for i, impl in enumerate(best_implementations, 1):
            profile_data = impl.get('profile', float('inf'))

            # 处理profile信息，支持三元组格式
            if isinstance(profile_data, (list, tuple)) and len(profile_data) >= 3:
                gen_time, base_time, speedup = profile_data[0], profile_data[1], profile_data[2]
                profile_str = f"生成代码: {gen_time:.4f}us, 基准代码: {base_time:.4f}us, 加速比: {speedup:.2f}x"
            elif isinstance(profile_data, (list, tuple)) and len(profile_data) >= 1:
                profile_str = f"执行时间: {profile_data[0]:.4f}us"
            elif profile_data != float('inf'):
                profile_str = f"执行时间: {profile_data:.4f}us"
            else:
                profile_str = "性能: N/A"

            # 构建显示信息
            info_parts = [f"{impl.get('op_name', 'Unknown')} (轮次 {impl.get('round', 'N/A')}"]

            # 只有在启用岛屿模型时才显示来源岛屿信息
            if evolution_result.get('island_info', {}).get('num_islands', 1) > 1:
                source_island = impl.get('source_island', 'N/A')
                info_parts.append(f"来源岛屿 {source_island}")

            info_parts.append(profile_str)
            print(f"  {i}. {', '.join(info_parts)})")
    else:
        print("\n⚠️  没有找到成功的实现")

    # 显示每轮详细结果
    round_results = evolution_result.get('round_results', [])
    if round_results:
        print(f"\n每轮详细结果:")
        for round_result in round_results:
            round_num = round_result.get('round', 'N/A')
            success_rate = round_result.get('success_rate', 0.0)
            successful = round_result.get('successful_tasks', 0)
            total = round_result.get('total_tasks', 0)
            print(f"  轮次 {round_num}: {successful}/{total} 成功 ({success_rate:.2%})")

    print("="*80)

    # 保存结果到文件
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M")
    file_name = f"evolve_result_{evolution_result.get('op_name', 'unknown')}_{evolve_config.dsl}_{evolve_config.framework}_{timestamp_str}.json"
    result_file = Path(config.get("log_dir", "")) / file_name

    # 为了JSON序列化，需要处理可能包含不可序列化对象的task_info字段
    serializable_result = evolution_result.copy()
    if 'best_implementations' in serializable_result:
        serializable_implementations = []
        for impl in serializable_result['best_implementations']:
            serializable_impl = impl.copy()
            # 从task_info中提取关键代码信息，然后移除整个task_info字段
            if 'task_info' in serializable_impl:
                task_info = serializable_impl['task_info']
                # 提取关键代码字段
                serializable_impl['designer_code'] = task_info.get('designer_code', '')
                serializable_impl['coder_code'] = task_info.get('coder_code', '')
                serializable_impl['task_desc'] = task_info.get('task_desc', '')
                serializable_impl['verifier_result'] = task_info.get('verifier_result', False)
                serializable_impl['verifier_error'] = task_info.get('verifier_error', '')
                # 移除复杂的task_info对象
                del serializable_impl['task_info']

            # 确保profile三元组可以JSON序列化
            if 'profile' in serializable_impl and isinstance(serializable_impl['profile'], tuple):
                serializable_impl['profile'] = list(serializable_impl['profile'])
            serializable_implementations.append(serializable_impl)
        serializable_result['best_implementations'] = serializable_implementations

    # 处理round_results中的implementations
    if 'round_results' in serializable_result:
        serializable_rounds = []
        for round_result in serializable_result['round_results']:
            serializable_round = round_result.copy()
            if 'implementations' in serializable_round:
                serializable_impls = []
                for impl in serializable_round['implementations']:
                    serializable_impl = impl.copy()
                    # 从task_info中提取关键代码信息，然后移除整个task_info字段
                    if 'task_info' in serializable_impl:
                        task_info = serializable_impl['task_info']
                        # 提取关键代码字段
                        serializable_impl['designer_code'] = task_info.get('designer_code', '')
                        serializable_impl['coder_code'] = task_info.get('coder_code', '')
                        serializable_impl['task_desc'] = task_info.get('task_desc', '')
                        serializable_impl['verifier_result'] = task_info.get('verifier_result', False)
                        serializable_impl['verifier_error'] = task_info.get('verifier_error', '')
                        # 移除复杂的task_info对象
                        del serializable_impl['task_info']

                    # 确保profile三元组可以JSON序列化
                    if 'profile' in serializable_impl and isinstance(serializable_impl['profile'], tuple):
                        serializable_impl['profile'] = list(serializable_impl['profile'])
                    serializable_impls.append(serializable_impl)
                serializable_round['implementations'] = serializable_impls
            serializable_rounds.append(serializable_round)
        serializable_result['round_results'] = serializable_rounds

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {result_file}")

    return evolution_result


def main():
    """主函数"""
    # 解析命令行参数
    project_root = get_project_root()
    config_path = os.path.join(project_root, "config", "evolve_config.yaml")
    if len(sys.argv) == 1:
        # 无参数模式：使用默认配置文件
        try:
            config = EvolveConfig.from_yaml(config_path)
            op_name = config.op_name
            task_desc = config.task_desc

            print(f"使用默认配置文件: {config_path}")
            print(f"算子名称: {op_name}")
            print(f"任务描述文件: {task_desc}")
            print(f"配置详情: {config.to_dict()}")
        except Exception as e:
            print(f"无法加载默认配置文件 {config_path}: {e}")
            print("使用内置默认配置")
            config = EvolveConfig()
            op_name = config.op_name
            task_desc = config.task_desc
            
            print(f"算子名称: {op_name}")
            print(f"任务描述文件: {task_desc}")

    elif len(sys.argv) == 2:
        # 配置文件模式：从YAML配置文件加载
        config_path = sys.argv[1]
        try:
            config = EvolveConfig.from_yaml(config_path)
            op_name = config.op_name
            task_desc = config.task_desc

            print(f"使用配置文件: {config_path}")
            print(f"算子名称: {op_name}")
            print(f"任务描述文件: {task_desc}")
            print(f"配置详情: {config.to_dict()}")
        except Exception as e:
            print(f"无法加载配置文件 {config_path}: {e}")
            sys.exit(1)

    elif len(sys.argv) > 3:
        # batch_runner简化模式: op_name task_file device [config_file]
        op_name = sys.argv[1]
        task_file = sys.argv[2]
        device = int(sys.argv[3])

        # 创建配置对象
        config = EvolveConfig()
        
        # 如果提供了配置文件路径
        if len(sys.argv) == 5:
            config_path = sys.argv[4]

        try:
            # 批量调用模式：跳过任务配置，因为任务文件是直接传入的
            file_config = EvolveConfig.from_yaml(config_path, skip_task_config=True)
            # 合并配置
            for key, value in file_config.to_dict().items():
                setattr(config, key, value)
            
            # 检查是否有自定义任务配置
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f)
                
                # 检查是否有custom_tasks配置
                if 'custom_tasks' in yaml_config and yaml_config['custom_tasks']:
                    if op_name in yaml_config['custom_tasks']:
                        custom_config = yaml_config['custom_tasks'][op_name]
                        print(f"🎯 发现自定义配置 for {op_name}: {custom_config}")
                        
                        # 应用自定义配置
                        if 'max_rounds' in custom_config:
                            config.max_rounds = custom_config['max_rounds']
                            print(f"   自定义 max_rounds: {config.max_rounds}")
                        if 'parallel_num' in custom_config:
                            config.parallel_num = custom_config['parallel_num']
                            print(f"   自定义 parallel_num: {config.parallel_num}")
                        if 'num_islands' in custom_config:
                            config.num_islands = custom_config['num_islands']
                            print(f"   自定义 num_islands: {config.num_islands}")
                        if 'migration_interval' in custom_config:
                            config.migration_interval = custom_config['migration_interval']
                            print(f"   自定义 migration_interval: {config.migration_interval}")
                        if 'elite_size' in custom_config:
                            config.elite_size = custom_config['elite_size']
                            print(f"   自定义 elite_size: {config.elite_size}")
                        if 'parent_selection_prob' in custom_config:
                            config.parent_selection_prob = custom_config['parent_selection_prob']
                            print(f"   自定义 parent_selection_prob: {config.parent_selection_prob}")
                        
                        print(f"✅ 已应用自定义配置")
                    
            except Exception as e:
                print(f"提示: 无法解析custom_tasks配置: {e}")
                
        except Exception as e:
            print(f"警告: 无法加载配置文件 {config_path}: {e}")

        # 设置设备
        config.device_list = [device]

        # 读取任务描述文件
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_desc = f.read().strip()

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

        except FileNotFoundError:
            print(f"任务文件不存在: {task_file}")
            sys.exit(1)
        except Exception as e:
            print(f"读取任务文件失败: {e}")
            sys.exit(1)

    else:
        print("用法:")
        print("  python single_evolve_runner.py                                                                        # 使用默认配置")
        print("  python single_evolve_runner.py <config_file>                                                          # 使用YAML配置文件")
        print(
            "  python single_evolve_runner.py <op_name> <task_file> <device> [config_file]                           # batch runner简化模式")
        sys.exit(1)

    # 运行任务
    try:
        result = asyncio.run(run_custom_evolve(op_name=op_name, task_desc=task_desc, evolve_config=config))

        if result:
            print("\n🎉 进化式算子生成成功完成!")
            successful_tasks = result.get('successful_tasks', 0)
            if successful_tasks > 0:
                print(f"✅ 成功生成了 {successful_tasks} 个有效的算子实现")
            else:
                print("⚠️  未能生成成功的算子实现，请检查配置和任务描述")
        else:
            print("\n❌ 进化过程失败，请检查日志获取详细信息")

    except Exception as e:
        print(f"Error occurred during evolution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()