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
自适应搜索示例 - Triton CUDA

使用基于 UCB 选择策略的自适应搜索框架生成 Triton CUDA Kernel。

特点：
- 异步流水线：任务完成立即补充，无等待浪费
- UCB 选择：智能平衡性能利用与探索
- 父代+灵感：选择一个父代 + 层次化采样其他灵感
- Sketch 更新：根据最终代码重新生成 sketch

用法:
  python run_torch_adaptive_search_triton_cuda.py
  
  # 自定义参数
  python run_torch_adaptive_search_triton_cuda.py --max-tasks 30 --max-concurrent 4
"""

import asyncio
import argparse

from akg_agents.op.adaptive_search import adaptive_search
from akg_agents.core.worker.manager import register_worker
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task
from akg_agents import get_project_root
from pathlib import Path


def get_op_name():
    """返回算子名称"""
    return 'akg_agents_relu'


def get_task_desc():
    """返回任务描述（PyTorch + CUDA 模型代码）"""
    return '''
import torch
import torch.nn as nn


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
    x = torch.rand(batch_size, dim, device='cuda')
    return [x]

def get_init_inputs():
    return []
'''


def print_config(args):
    """打印配置信息"""
    print("=" * 60)
    print("自适应搜索配置 (Triton CUDA)")
    print("=" * 60)
    print(f"算子名称: {get_op_name()}")
    print(f"DSL: triton_cuda")
    print(f"框架: torch")
    print(f"后端: cuda")
    print(f"架构: rtx3090")
    print("-" * 60)
    print(f"设备列表: {args.devices}")
    print(f"最大并发数: {args.max_concurrent}")
    print(f"初始任务数: {args.initial_tasks}")
    print(f"最大总任务数: {args.max_tasks}")
    print("-" * 60)
    print(f"UCB 探索系数: {args.exploration_coef}")
    print(f"随机扰动: {args.random_factor}")
    print(f"灵感采样: 父代 + {args.inspiration_num}个（层次化采样）")
    print("=" * 60)


def print_result(result):
    """打印搜索结果"""
    import os
    
    print("\n" + "=" * 100)
    print("自适应搜索结果")
    print("=" * 100)
    
    print(f"算子名称：{result['op_name']}")
    print(f"终止原因：{result.get('stop_reason', 'Unknown')}")
    print(f"任务统计：提交{result['total_submitted']} / 完成{result.get('total_completed', 0)} / 成功{result['total_success']} / 失败{result['total_failed']} | 成功率{result['success_rate']:.1%} | 耗时{result['elapsed_time']:.1f}s")
    print(f"存储目录：{result.get('storage_dir', 'N/A')}")
    
    task_folder = result.get('task_folder', '')
    if task_folder:
        print(f"Task文件夹：{task_folder}")
    
    log_dir = result.get('log_dir', '')
    if log_dir:
        print(f"Log目录：{log_dir}")
    
    lineage_graph = result.get('lineage_graph', '')
    if lineage_graph:
        print(f"谱系图：{lineage_graph}")
    
    print("\n最佳实现（前5个）：")
    
    best_impls = result.get('best_implementations', [])
    if best_impls:
        for i, impl in enumerate(best_impls[:5], 1):
            task_id = impl.get('id', 'unknown')
            gen_time = impl.get('gen_time', 0)
            profile = impl.get('profile', {})
            base_time = profile.get('base_time', 0) if profile else 0
            speedup = impl.get('speedup', 0)
            generation = impl.get('generation', 0)
            parent_id = impl.get('parent_id', None)
            verify_dir = impl.get('verify_dir', '')
            
            if generation == 0:
                parent_desc = "初始"
            else:
                parent_desc = f"父代 {parent_id}" if parent_id else f"G{generation}"
            
            print(f"{i}. {task_id}（{parent_desc}，个体路径：{verify_dir}，生成代码：{gen_time:.4f}us，基准代码：{base_time:.4f}us，加速比：{speedup:.2f}x）")
    else:
        print("未找到成功的实现")
    
    print("\n" + "=" * 100)


async def run_adaptive_search(args):
    """运行自适应搜索"""
    
    print_config(args)
    
    dsl = "triton_cuda"
    framework = "torch"
    backend = "cuda"
    arch = "rtx3090"
    
    config_path = str(Path(get_project_root()) / "config" / "triton_cuda_evolve_config.yaml")
    
    print(f"\n{'='*60}")
    print("注册 Worker")
    print(f"{'='*60}")
    print(f"🔗 注册 Worker: devices={args.devices}")
    await register_worker(
        backend=backend,
        arch=arch,
        device_ids=args.devices
    )
    
    try:
        loaded_config = load_config(config_path=config_path)
    except ValueError:
        print(f"Config file {config_path} not found, using default.")
        loaded_config = load_config(dsl=dsl, backend=backend)
    
    from akg_agents.utils.task_label import resolve_task_label
    loaded_config["task_label"] = resolve_task_label(
        op_name=get_op_name(),
        parallel_index=1,
    )
    
    check_env_for_task(framework, backend, dsl, loaded_config)
    
    print(f"\n{'='*60}")
    print("开始自适应搜索...")
    print(f"{'='*60}\n")
    
    result = await adaptive_search(
        op_name=get_op_name(),
        task_desc=get_task_desc(),
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        config=loaded_config,
        
        max_concurrent=args.max_concurrent,
        initial_task_count=args.initial_tasks,
        
        exploration_coef=args.exploration_coef,
        random_factor=args.random_factor,
        
        max_total_tasks=args.max_tasks,
        
        inspiration_sample_num=args.inspiration_num,
        use_tiered_sampling=True,
        handwrite_sample_num=2,
        handwrite_decay_rate=2.0
    )
    
    print_result(result)
    
    return result


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="运行自适应搜索示例 (Triton CUDA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认配置
  python run_torch_adaptive_search_triton_cuda.py
  
  # 使用多个 GPU 设备
  python run_torch_adaptive_search_triton_cuda.py --devices 0 1 2 3 --max-concurrent 4
  
  # 自定义参数
  python run_torch_adaptive_search_triton_cuda.py --max-tasks 30
  
  # 更多探索
  python run_torch_adaptive_search_triton_cuda.py --exploration-coef 2.0
        """
    )
    
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=[4, 5, 6, 7],
        help="GPU 设备 ID 列表，默认: [4, 5, 6, 7]"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="最大并发任务数，默认: 4"
    )
    parser.add_argument(
        "--initial-tasks",
        type=int,
        default=4,
        help="初始任务数量，默认: 4"
    )
    
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=20,
        help="最大总任务数，默认: 20"
    )
    parser.add_argument(
        "--exploration-coef",
        type=float,
        default=1.414,
        help="UCB 探索系数 (默认: 1.414)"
    )
    parser.add_argument(
        "--random-factor",
        type=float,
        default=0.1,
        help="选择时的随机扰动 (0~1)，默认: 0.1"
    )
    
    parser.add_argument(
        "--inspiration-num",
        type=int,
        default=3,
        help="灵感采样数量（不含父代），默认: 3"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("自适应搜索示例 - Triton CUDA")
    print("=" * 60)
    
    asyncio.run(run_adaptive_search(args))


if __name__ == "__main__":
    main()
