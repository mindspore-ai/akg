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
from ai_kernel_generator.core.evolve import evolve
from ai_kernel_generator.core.async_pool.task_pool import TaskPool
from ai_kernel_generator.core.async_pool.device_pool import DevicePool
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.database.database import Database


def get_op_name():
    """获取算子名称"""
    return "aikg_relu"


def get_task_desc():
    """获取任务描述"""
    return '''
import torch
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
    return []  # No special initialization inputs needed
'''


async def run_evolve_example():
    """运行进化式算子生成示例"""
    # 基本参数配置
    op_name = get_op_name()
    task_desc = get_task_desc()
    dsl = "triton"  # 可选: "triton", "swft"
    framework = "torch"  # 可选: "mindspore", "torch", "numpy"
    backend = "ascend"  # 可选: "ascend", "cuda"
    arch = "ascend910b4"  # 根据backend选择对应架构

    # 进化参数配置
    max_rounds = 3  # 进化轮数
    parallel_num = 4  # 每轮并行任务数

    print("="*80)
    print("AI KERNEL GENERATOR - 进化式算子生成示例")
    print("="*80)
    print(f"算子名称: {op_name}")
    print(f"实现类型: {dsl}")
    print(f"框架: {framework}")
    print(f"后端: {backend}")
    print(f"架构: {arch}")
    print(f"进化轮数: {max_rounds}")
    print(f"并行任务数: {parallel_num}")
    print("="*80)

    # 初始化资源
    task_pool = TaskPool(max_concurrency=parallel_num)
    device_pool = DevicePool([0, 1])  # 使用设备0和1

    # 运行进化过程
    print("开始进化过程...")
    evolution_result = await evolve(
        op_name=op_name,
        task_desc=task_desc,
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        config=load_config(dsl),
        device_pool=device_pool,
        task_pool=task_pool,
        max_rounds=max_rounds,
        parallel_num=parallel_num
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

    # 显示最佳实现
    best_implementations = evolution_result.get('best_implementations', [])
    if best_implementations:
        print(f"\n最佳实现 (前{len(best_implementations)}个):")
        for i, impl in enumerate(best_implementations, 1):
            profile_score = impl.get('profile', float('inf'))
            profile_str = f"性能: {profile_score:.4f}" if profile_score != float('inf') else "性能: N/A"
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
            print(f"  轮次 {round_num}: {successful}/{total} 成功 ({success_rate:.2%})")

    print("="*80)

    # 保存结果到文件
    import json
    result_file = f"evolve_result_{evolution_result.get('op_name', 'unknown')}_{dsl}_{framework}.json"

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
    # 运行异步进化过程
    result = asyncio.run(run_evolve_example())

    if result:
        print("\n🎉 进化式算子生成成功完成!")
        successful_tasks = result.get('successful_tasks', 0)
        if successful_tasks > 0:
            print(f"✅ 成功生成了 {successful_tasks} 个有效的算子实现")
        else:
            print("⚠️  未能生成成功的算子实现，请检查配置和任务描述")
    else:
        print("\n❌ 进化过程失败，请检查日志获取详细信息")


if __name__ == "__main__":
    main()
