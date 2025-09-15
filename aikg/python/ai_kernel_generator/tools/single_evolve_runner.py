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
from pathlib import Path
import json
from datetime import datetime
from ai_kernel_generator.core.evolve import evolve
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task
from ai_kernel_generator import get_project_root
from typing import Optional, List
from dataclasses import dataclass

# ============================================================================
# 配置参数类
# ============================================================================


@dataclass
class EvolveConfig:
    """进化配置参数类"""
    # 基本参数
    dsl: str = "triton"
    framework: str = "torch"
    backend: str = "ascend"
    arch: str = "ascend910b4"

    # 进化参数
    max_rounds: int = 2
    parallel_num: int = 2

    # 设备配置
    device_list: List[int] = None

    # 配置文件路径
    config_path: str = ""  # 将在__post_init__中设置

    # 任务配置
    op_name: Optional[str] = None
    task_desc: Optional[str] = None

    def __post_init__(self):
        if self.device_list is None:
            self.device_list = [5]
        if not self.config_path:
            self.config_path = str(Path(get_project_root()) / "config" / "vllm_triton_evolve_config.yaml")


# 默认配置实例 - 在此处修改基础配置
DEFAULT_CONFIG = EvolveConfig(
    dsl="triton",
    framework="torch",
    backend="ascend",
    arch="ascend910b4",
    max_rounds=2,
    parallel_num=2,
    device_list=[5],
    config_path=""  # 将在__post_init__中自动设置
)

# 默认任务配置
DEFAULT_OP_NAME = "relu_op"
DEFAULT_TASK_DESC = """import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []
"""


async def run_custom_evolve(op_name: str = None, task_desc: str = None, evolve_config: EvolveConfig = None):
    """运行自定义任务的进化过程

    Args:
        op_name: 算子名称，如果为None则使用evolve_config中的配置
        task_desc: 任务描述，如果为None则使用evolve_config中的配置
        evolve_config: 进化配置类实例，如果为None则使用默认配置
    """
    # 使用传入的配置或默认配置
    if evolve_config is None:
        evolve_config = DEFAULT_CONFIG

    # 如果op_name或task_desc为None，尝试从配置中获取
    if op_name is None:
        op_name = evolve_config.op_name or DEFAULT_OP_NAME
    if task_desc is None:
        task_desc = evolve_config.task_desc or DEFAULT_TASK_DESC.strip()

    print("="*80)
    print("AI KERNEL GENERATOR - 进化式算子生成示例")
    print("="*80)
    print(f"算子名称: {op_name}")
    print(f"实现类型: {evolve_config.dsl}")
    print(f"框架: {evolve_config.framework}")
    print(f"后端: {evolve_config.backend}")
    print(f"架构: {evolve_config.arch}")
    print(f"进化轮数: {evolve_config.max_rounds}")
    print(f"并行任务数: {evolve_config.parallel_num}")
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
        parallel_num=evolve_config.parallel_num
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
                profile_str = f"生成代码: {gen_time:.4f}s, 基准代码: {base_time:.4f}s, 加速比: {speedup:.2f}x"
            elif isinstance(profile_data, (list, tuple)) and len(profile_data) >= 1:
                profile_str = f"执行时间: {profile_data[0]:.4f}s"
            elif profile_data != float('inf'):
                profile_str = f"执行时间: {profile_data:.4f}s"
            else:
                profile_str = "性能: N/A"

            print(f"  {i}. {impl.get('op_name', 'Unknown')} (轮次 {impl.get('round', 'N/A')}, {profile_str})")
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
            round_best_speedup = round_result.get('round_best_speedup', 0.0)
            global_best_speedup = round_result.get('global_best_speedup', 0.0)

            print(f"  轮次 {round_num}: {successful}/{total} 成功 ({success_rate:.2%}), "
                  f"本轮最佳: {round_best_speedup:.2f}x, 全局最佳: {global_best_speedup:.2f}x")

    # 显示加速比统计汇总
    round_best_speedups = evolution_result.get('round_best_speedups', [])
    global_best_speedup_history = evolution_result.get('global_best_speedup_history', [])
    final_best_speedup = evolution_result.get('final_best_speedup', 0.0)

    if round_best_speedups:
        print(f"\n🚀 加速比统计汇总:")
        print(f"  每轮最佳加速比: {[f'{x:.2f}x' for x in round_best_speedups]}")
        print(f"  截至每轮全局最佳: {[f'{x:.2f}x' for x in global_best_speedup_history]}")
        print(f"  最终全局最佳加速比: {final_best_speedup:.2f}x")

        # 计算加速比改进趋势
        if len(round_best_speedups) > 1:
            improvements = []
            for i in range(1, len(round_best_speedups)):
                if round_best_speedups[i] > round_best_speedups[i-1]:
                    improvements.append(f"轮次{i+1}")
            if improvements:
                print(f"  性能改进轮次: {', '.join(improvements)}")

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
    if len(sys.argv) == 1:
        # 无参数模式：使用硬编码的默认配置
        op_name = DEFAULT_OP_NAME
        task_desc = DEFAULT_TASK_DESC.strip()
        config = DEFAULT_CONFIG

        print(f"使用默认硬编码任务: {op_name}")
        print("任务描述: 使用内置默认任务")

    elif len(sys.argv) == 10:
        # batch_runner调用模式: op_name task_file device max_rounds parallel_num dsl framework backend arch
        op_name = sys.argv[1]
        task_file = sys.argv[2]
        device = int(sys.argv[3])
        max_rounds = int(sys.argv[4])
        parallel_num = int(sys.argv[5])
        dsl = sys.argv[6]
        framework = sys.argv[7]
        backend = sys.argv[8]
        arch = sys.argv[9]

        # 创建配置对象
        config = EvolveConfig(
            dsl=dsl,
            framework=framework,
            backend=backend,
            arch=arch,
            max_rounds=max_rounds,
            parallel_num=parallel_num,
            device_list=[device],
            config_path=""  # 将在__post_init__中自动设置
        )

        # 读取任务描述文件
        try:
            with open(task_file, 'r', encoding='utf-8') as f:
                task_desc = f.read().strip()

            print(f"批量模式任务: {op_name}")
            print(f"任务文件: {task_file}")
            print(f"设备: {config.device_list}")
            print(f"配置: {config.max_rounds}轮/{config.parallel_num}并行")
            print(f"基础参数: {config.dsl}/{config.framework}/{config.backend}/{config.arch}")

        except FileNotFoundError:
            print(f"任务文件不存在: {task_file}")
            sys.exit(1)
        except Exception as e:
            print(f"读取任务文件失败: {e}")
            sys.exit(1)

    else:
        print("用法:")
        print("  python run_batch_evolve.py                                                                        # 使用硬编码默认配置")
        print("  python run_batch_evolve.py <op_name> <task_file> <device> <rounds> <parallel> <dsl> <framework> <backend> <arch>  # batch模式")
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
        sys.exit(1)


if __name__ == "__main__":
    main()
