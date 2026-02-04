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
自适应搜索测试 - Triton Ascend 910B2

测试配置：
- 默认并发数：3
- 最大任务数：10
- 架构：ascend910b2
- 可查看性能数据（加速比、执行时间等）
"""

import os
import pytest
from pathlib import Path

# 注意：不在模块顶部导入 adaptive_search，避免循环导入问题
# adaptive_search 会在测试函数内部延迟导入

from ai_kernel_generator.core.worker.manager import register_local_worker
from ai_kernel_generator.config.config_validator import load_config
from ai_kernel_generator.utils.environment_check import check_env_for_task
from ..utils import (
    get_kernelbench_op_name, get_kernelbench_task_desc,
    add_op_prefix, get_device_id
)

os.environ['AIKG_DATA_COLLECT'] = 'off'
device_id = get_device_id()

# ============================================================================
# 默认测试参数
# ============================================================================
DEFAULT_MAX_CONCURRENT = 3      # 默认最大并发数
DEFAULT_MAX_TASKS = 10           # 默认最大任务数
DEFAULT_INITIAL_TASKS = 3       # 默认初始任务数
DEFAULT_ARCH = "ascend910b2"    # 默认架构


def print_adaptive_search_result(result):
    """
    打印自适应搜索结果，包含性能数据

    Args:
        result: adaptive_search 返回的结果字典
    """
    print("\n" + "=" * 100)
    print("自适应搜索结果")
    print("=" * 100)

    print(f"算子名称：{result['op_name']}")
    print(f"终止原因：{result.get('stop_reason', 'Unknown')}")
    print(f"任务统计：提交{result['total_submitted']} / 完成{result.get('total_completed', 0)} / "
          f"成功{result['total_success']} / 失败{result['total_failed']} | "
          f"成功率{result['success_rate']:.1%} | 耗时{result['elapsed_time']:.1f}s")
    print(f"存储目录：{result.get('storage_dir', 'N/A')}")

    # 打印 Task 文件夹和 Log 目录
    task_folder = result.get('task_folder', '')
    if task_folder:
        print(f"Task文件夹：{task_folder}")

    log_dir = result.get('log_dir', '')
    if log_dir:
        print(f"Log目录：{log_dir}")

    # 打印谱系图路径
    lineage_graph = result.get('lineage_graph', '')
    if lineage_graph:
        print(f"谱系图：{lineage_graph}")

    # 打印最佳实现
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

            # 父代描述
            if generation == 0:
                parent_desc = "初始"
            else:
                parent_desc = f"父代 {parent_id}" if parent_id else f"G{generation}"

            # 格式：序号. 任务ID（父代信息，个体路径：xxx，生成代码：xxxus，基准代码：xxxus，加速比：x.xx）
            print(f"{i}. {task_id}（{parent_desc}，个体路径：{verify_dir}，"
                  f"生成代码：{gen_time:.4f}us，基准代码：{base_time:.4f}us，加速比：{speedup:.2f}x）")
    else:
        print("未找到成功的实现")

    print("\n" + "=" * 100)

    return result


def generate_adaptive_search_report(result, config, framework, dsl, backend, arch,
                                     save_to_file=True, output_dir=None):
    """
    生成自适应搜索性能报告

    Args:
        result: adaptive_search 返回的结果字典
        config: 配置字典
        framework: 框架名称
        dsl: DSL名称
        backend: 后端名称
        arch: 架构名称
        save_to_file: 是否保存到文件
        output_dir: 输出目录
    """
    # 确定输出目录
    if output_dir is None:
        result_dir = Path(os.path.expanduser(config['log_dir']))
    else:
        result_dir = Path(output_dir)

    # 控制台输出
    print_adaptive_search_result(result)

    # 保存到文件
    if save_to_file:
        result_dir.mkdir(parents=True, exist_ok=True)
        report_path = result_dir / "adaptive_search_report.txt"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=" * 100 + "\n")
            f.write("自适应搜索结果\n")
            f.write("=" * 100 + "\n\n")

            f.write(f"算子名称：{result['op_name']}\n")
            f.write(f"终止原因：{result.get('stop_reason', 'Unknown')}\n")
            f.write(f"任务统计：提交{result['total_submitted']} / 完成{result.get('total_completed', 0)} / "
                    f"成功{result['total_success']} / 失败{result['total_failed']} | "
                    f"成功率{result['success_rate']:.1%} | 耗时{result['elapsed_time']:.1f}s\n")
            f.write(f"存储目录：{result.get('storage_dir', 'N/A')}\n")

            task_folder = result.get('task_folder', '')
            if task_folder:
                f.write(f"Task文件夹：{task_folder}\n")

            log_dir = result.get('log_dir', '')
            if log_dir:
                f.write(f"Log目录：{log_dir}\n")

            lineage_graph = result.get('lineage_graph', '')
            if lineage_graph:
                f.write(f"谱系图：{lineage_graph}\n")

            f.write("\n最佳实现（前5个）：\n")

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

                    f.write(f"{i}. {task_id}（{parent_desc}，个体路径：{verify_dir}，"
                            f"生成代码：{gen_time:.4f}us，基准代码：{base_time:.4f}us，加速比：{speedup:.2f}x）\n")
            else:
                f.write("未找到成功的实现\n")

            f.write("\n" + "=" * 100 + "\n")

        print(f"性能报告已保存到 {report_path}")

    return {
        'op_name': result['op_name'],
        'total_submitted': result['total_submitted'],
        'total_success': result['total_success'],
        'total_failed': result['total_failed'],
        'success_rate': result['success_rate'],
        'elapsed_time': result['elapsed_time'],
        'best_implementations': result.get('best_implementations', [])
    }


# ============================================================================
# 测试用例
# ============================================================================

@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b2
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_adaptive_search_kernelbench_torch_triton_ascend910b2():
    """
    自适应搜索测试 - KernelBench 算子 (PyTorch + Triton Ascend 910B2)

    测试配置:
    - 并发数: 3
    - 最大任务数: 10
    - 架构: ascend910b2
    - 测试 KernelBench 第23号算子
    """
    # 延迟导入，避免循环导入问题
    from ai_kernel_generator.core.adaptive_search import adaptive_search

    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = DEFAULT_ARCH
    benchmark = "KernelBench"

    # 加载配置
    config = load_config(
        config_path="./python/ai_kernel_generator/config/vllm_triton_ascend_evolve_config.yaml"
    )

    check_env_for_task(framework, backend, dsl, config)

    # 注册 LocalWorker
    await register_local_worker([device_id], backend=backend, arch=arch)

    # 获取 KernelBench 算子
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[23], framework=framework
    )

    if benchmark_name is None or len(benchmark_name) == 0:
        pytest.skip(f"benchmark '{benchmark}' 不支持或算子不存在")

    # 获取第一个算子的任务描述
    task_desc = get_kernelbench_task_desc(benchmark_name[0], framework=framework)
    op_name = add_op_prefix(benchmark_name[0], benchmark=benchmark) + "_adaptive"

    print("=" * 60)
    print(f"🚀 自适应搜索测试 - {op_name}")
    print("=" * 60)
    print(f"   基准测试: {benchmark}")
    print(f"   框架: {framework}")
    print(f"   DSL: {dsl}")
    print(f"   后端: {backend}")
    print(f"   架构: {arch}")
    print(f"   并发数: {DEFAULT_MAX_CONCURRENT}")
    print(f"   最大任务数: {DEFAULT_MAX_TASKS}")
    print("=" * 60)

    # 运行自适应搜索
    result = await adaptive_search(
        op_name=op_name,
        task_desc=task_desc,
        dsl=dsl,
        framework=framework,
        backend=backend,
        arch=arch,
        config=config,

        # 并发控制
        max_concurrent=DEFAULT_MAX_CONCURRENT,
        initial_task_count=DEFAULT_INITIAL_TASKS,

        # 停止条件
        max_total_tasks=DEFAULT_MAX_TASKS,

        # UCB 选择参数
        exploration_coef=1.414,
        random_factor=0.1,

        # 灵感采样参数
        inspiration_sample_num=3,
        use_tiered_sampling=True,
        handwrite_sample_num=2,
        handwrite_decay_rate=2.0
    )

    # 生成性能报告
    generate_adaptive_search_report(
        result, config, framework, dsl, backend, arch
    )

    # 验证结果
    assert result is not None, "自适应搜索返回结果为空"
    assert result['total_submitted'] > 0, "未提交任何任务"