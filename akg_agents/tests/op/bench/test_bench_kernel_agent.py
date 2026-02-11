"""KernelAgent 批量测试

使用 KernelAgent（ReAct Agent）批量运行 Benchmark 任务。
与 test_bench_triton_ascend.py 等使用 LangGraphTask 的测试不同，
本模块直接使用 KernelAgent 的对话式接口，以自动模式运行。

使用示例:
  # 运行 KernelBench (torch + triton_ascend)
  pytest tests/op/bench/test_bench_kernel_agent.py::test_kernelbench_torch_triton_ascend910b4 -v

  # 运行 KernelBench (torch + triton_cuda)
  pytest tests/op/bench/test_bench_kernel_agent.py::test_kernelbench_torch_triton_cuda -v

  # 运行 KernelBench (torch + cpp)
  pytest tests/op/bench/test_bench_kernel_agent.py::test_kernelbench_torch_cpp_cpu -v

  # 运行自定义需求批量测试
  pytest tests/op/bench/test_bench_kernel_agent.py::test_custom_requirements_kernel_agent -v
"""

import os
import time
import logging
import pytest
from pathlib import Path
from collections import defaultdict

from akg_agents.op.agents.kernel_agent import KernelAgent
from akg_agents.core.async_pool.task_pool import TaskPool
from akg_agents.core.worker.manager import register_local_worker
from ..utils import (
    get_kernelbench_op_name, get_multikernelbench_op_name,
    get_kernelbench_task_desc, get_multikernelbench_task_desc,
    get_akg_kernels_bench_op_name, get_akg_kernels_bench_task_desc,
    add_op_prefix, generate_beautiful_test_report, get_device_id
)
from akg_agents.op.config.config_validator import load_config
from akg_agents.utils.environment_check import check_env_for_task

logger = logging.getLogger(__name__)

os.environ['AKG_AGENTS_DATA_COLLECT'] = 'on'
device_id = get_device_id()

# ==================== 自动模式配置 ====================

# Agent 流程中需要确认时的默认回复
AUTO_REPLY_MESSAGE = "同意你的方案，请直接继续执行，不需要再确认。初始需求完成后直接结束任务，不需要进一步优化。"

# 单个任务最大轮次（防止死循环）
AUTO_MAX_ROUNDS = 50


# ==================== 核心执行函数 ====================

async def run_single_kernel_agent(
    op_name: str,
    task_desc: str,
    framework: str = "torch",
    backend: str = "ascend",
    arch: str = "ascend910b4",
    dsl: str = "triton_ascend",
    model_level: str = "complex",
    max_rounds: int = AUTO_MAX_ROUNDS,
    auto_reply: str = AUTO_REPLY_MESSAGE,
    config: dict = None,
) -> tuple:
    """
    以自动模式运行单个 KernelAgent 任务直到完成。

    模拟 run_kernel_agent.py 中的 auto 模式循环：
    - 第 1 轮发送 task_desc 作为初始需求
    - 后续轮次对 ask_user 自动回复
    - task_completed=True 或 status=success/error 时结束

    Args:
        op_name: 算子名称（用于标识和返回）
        task_desc: 任务描述（即 benchmark 的代码内容）
        framework: 框架
        backend: 后端
        arch: 架构
        dsl: DSL
        model_level: 模型级别
        max_rounds: 最大轮次
        auto_reply: 自动回复内容
        config: 配置字典

    Returns:
        Tuple[str, bool, dict]: (op_name, 是否成功, 结果详情)
        与 LangGraphTask.run() 返回格式一致，便于复用 generate_beautiful_test_report
    """
    task_id = f"bench_{op_name}_{int(time.time())}"

    try:
        agent = KernelAgent(
            task_id=task_id,
            model_level=model_level,
            config=config,
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl,
        )

        user_input = task_desc
        round_num = 0
        final_result = {}

        while round_num < max_rounds:
            round_num += 1
            logger.info(f"[{op_name}] Round {round_num}/{max_rounds}")

            result = await agent.run(user_input)
            final_result = result
            status = result.get('status', 'unknown')

            if status == 'waiting_for_user':
                task_completed = result.get('task_completed', False)
                if task_completed:
                    logger.info(f"[{op_name}] 任务完成 (task_completed=True), 共 {round_num} 轮")
                    return op_name, True, result
                # 流程中的确认 → 自动回复
                logger.info(f"[{op_name}] 自动回复确认")
                user_input = auto_reply
                continue

            if status == 'success':
                logger.info(f"[{op_name}] 任务成功, 共 {round_num} 轮")
                return op_name, True, result

            if status == 'error':
                error_info = result.get('error_information', 'Unknown error')
                logger.warning(f"[{op_name}] 任务失败: {error_info}")
                return op_name, False, result

            # 未知状态，继续自动回复
            logger.info(f"[{op_name}] 未知状态 '{status}', 继续自动回复")
            user_input = auto_reply

        # 超出最大轮次
        logger.warning(f"[{op_name}] 超出最大轮次 {max_rounds}")
        return op_name, False, {
            "error": f"超出最大轮次 ({max_rounds})",
            **final_result
        }

    except Exception as e:
        logger.error(f"[{op_name}] 异常: {e}", exc_info=True)
        return op_name, False, {"error": str(e)}


# ==================== KernelBench 测试 ====================

@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_triton_ascend910b4():
    """KernelAgent 批量测试 - KernelBench (torch + triton_ascend + ascend910b4)"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    config = load_config(
        config_path="./python/akg_agents/op/config/vllm_triton_ascend_coderonly_config.yaml"
    )
    check_env_for_task(framework, backend, dsl, config)
    await register_local_worker([device_id], backend=backend, arch=arch)

    # KernelBench: 按序号读取，可修改 task_index_list 选择不同用例
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

    for i, name in enumerate(benchmark_name):
        task_desc = get_kernelbench_task_desc(name, framework=framework)
        op_name = add_op_prefix(name, benchmark=benchmark)

        task_pool.create_task(
            run_single_kernel_agent,
            op_name=op_name,
            task_desc=task_desc,
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            task_name=op_name,
        )

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )

@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_triton_cuda():
    """KernelAgent 批量测试 - KernelBench (torch + triton_cuda + a100)"""
    framework = "torch"
    dsl = "triton_cuda"
    backend = "cuda"
    arch = "a100"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    config = load_config(
        config_path="./python/akg_agents/op/config/vllm_triton_cuda_coderonly_config.yaml"
    )
    check_env_for_task(framework, backend, dsl, config)
    await register_local_worker([device_id], backend=backend, arch=arch)

    # KernelBench: 按序号读取，可修改 task_index_list 选择不同用例
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

    for i, name in enumerate(benchmark_name):
        task_desc = get_kernelbench_task_desc(name, framework=framework)
        op_name = add_op_prefix(name, benchmark=benchmark)

        task_pool.create_task(
            run_single_kernel_agent,
            op_name=op_name,
            task_desc=task_desc,
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            task_name=op_name,
        )

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )

@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_kernelbench_torch_cpp_cpu():
    """KernelAgent 批量测试 - KernelBench (torch + cpp + cpu)"""
    framework = "torch"
    dsl = "cpp"
    backend = "cpu"
    arch = "x86_64"
    benchmark = "KernelBench"

    task_pool = TaskPool()
    config = load_config(
        config_path="./python/akg_agents/op/config/vllm_cpp_coderonly_config.yaml"
    )
    check_env_for_task(framework, backend, dsl, config)
    await register_local_worker([device_id], backend=backend, arch=arch)

    # KernelBench: 按序号读取，可修改 task_index_list 选择不同用例
    benchmark_name = get_kernelbench_op_name(
        task_index_list=[19], framework=framework)

    if benchmark_name is None:
        raise RuntimeError(f"benchmark '{benchmark}' 不支持")

    for i, name in enumerate(benchmark_name):
        task_desc = get_kernelbench_task_desc(name, framework=framework)
        op_name = add_op_prefix(name, benchmark=benchmark)

        task_pool.create_task(
            run_single_kernel_agent,
            op_name=op_name,
            task_desc=task_desc,
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            task_name=op_name,
        )

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )


# ==================== 自定义需求批量测试 ====================

# 自定义需求列表：每项为 (名称, 需求描述)
# 可以根据需要修改或从外部文件加载
CUSTOM_REQUIREMENTS = [
    ("vector_add", "实现一个向量加法算子，支持 float16 和 float32 类型"),
    ("relu", "实现一个 ReLU 激活函数算子"),
    ("softmax", "实现一个 Softmax 算子，沿最后一个维度计算"),
]


@pytest.mark.level2
@pytest.mark.torch
@pytest.mark.triton
@pytest.mark.ascend
@pytest.mark.ascend910b4
@pytest.mark.use_model
@pytest.mark.asyncio
async def test_custom_requirements_kernel_agent():
    """KernelAgent 批量测试 - 自定义需求列表"""
    framework = "torch"
    dsl = "triton_ascend"
    backend = "ascend"
    arch = "ascend910b4"

    task_pool = TaskPool()
    config = load_config(
        config_path="./python/akg_agents/op/config/vllm_triton_ascend_coderonly_config.yaml"
    )
    check_env_for_task(framework, backend, dsl, config)
    await register_local_worker([device_id], backend=backend, arch=arch)

    requirements = _load_custom_requirements()

    for i, (name, desc) in enumerate(requirements):
        op_name = f"custom_{name}"
        task_pool.create_task(
            run_single_kernel_agent,
            op_name=op_name,
            task_desc=desc,
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl,
            config=config,
            task_name=op_name,
        )

    results = await task_pool.wait_all()

    report_stats = generate_beautiful_test_report(
        results, config, framework, dsl, backend, arch
    )


def _load_custom_requirements():
    """
    加载自定义需求列表。

    优先从环境变量 KERNEL_AGENT_REQUIREMENTS_FILE 指定的 JSON 文件加载，
    文件格式: [{"name": "xxx", "desc": "xxx"}, ...]

    若未指定则使用模块内的 CUSTOM_REQUIREMENTS 默认列表。
    """
    req_file = os.environ.get("KERNEL_AGENT_REQUIREMENTS_FILE")
    if req_file and os.path.exists(req_file):
        import json
        with open(req_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        return [(item["name"], item["desc"]) for item in items]
    return CUSTOM_REQUIREMENTS
