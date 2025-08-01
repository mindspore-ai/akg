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

    # 输出进化结果
    print("\n" + "="*80)
    print("进化完成！最终结果汇总:")
    print("="*80)
    print(f"算子名称: {evolution_result['op_name']}")
    print(f"总轮数: {evolution_result['total_rounds']}")
    print(f"总任务数: {evolution_result['total_tasks']}")
    print(f"成功任务数: {evolution_result['successful_tasks']}")
    print(f"最终成功率: {evolution_result['final_success_rate']:.2%}")
    print(f"最佳成功率: {evolution_result['best_success_rate']:.2%}")
    print(f"实现类型: {evolution_result['implementation_type']}")
    print(f"框架: {evolution_result['framework']}")
    print(f"后端: {evolution_result['backend']}")
    print(f"架构: {evolution_result['architecture']}")

    # 显示最佳实现
    best_implementations = evolution_result['best_implementations']
    if best_implementations:
        print(f"\n最佳实现 (前{len(best_implementations)}个):")
        for i, impl in enumerate(best_implementations, 1):
            print(f"  {i}. {impl['op_name']} (轮次 {impl['round']})")

    print("="*80)

    # 保存结果到文件
    import json
    result_file = f"evolve_result_{op_name}_{dsl}_{framework}.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(evolution_result, f, indent=2, ensure_ascii=False)
    print(f"结果已保存到: {result_file}")

    return evolution_result


def main():
    """主函数"""
    # 运行异步进化过程
    result = asyncio.run(run_evolve_example())

    if result:
        print("\n🎉 进化式算子生成成功完成!")
        if result['successful_tasks'] > 0:
            print(f"✅ 成功生成了 {result['successful_tasks']} 个有效的算子实现")
        else:
            print("⚠️  未能生成成功的算子实现，请检查配置和任务描述")
    else:
        print("\n❌ 进化过程失败，请检查日志获取详细信息")


if __name__ == "__main__":
    main()
