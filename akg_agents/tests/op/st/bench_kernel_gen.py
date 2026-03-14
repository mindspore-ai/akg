#!/usr/bin/env python3
# Copyright 2026 Huawei Technologies Co., Ltd
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
KernelGen 评测脚本（支持 pass@N，TaskPool 并发）

复用已有基础设施：TaskPool 并发 + generate_beautiful_test_report 报告。
每个算子独立生成 N 次（不重试、不走 conductor），评估：
1. py_compile 语法是否通过
2. KernelVerifier 正确性是否通过
3. pass@N：N 次中至少 1 次验证通过即算 pass

用法：
    python tests/st/bench_kernel_gen.py
    python tests/st/bench_kernel_gen.py --n 3                     # 每个算子跑 3 次
    python tests/st/bench_kernel_gen.py --indices 19,21,23 --n 5  # 指定算子，跑 5 次
    python tests/st/bench_kernel_gen.py --tag nojson              # 结果文件带 tag 区分
    python tests/st/bench_kernel_gen.py --concurrency 8           # 最大并发数
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import py_compile
import tempfile
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))
os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

_tmp_dir = os.path.join(os.path.expanduser("~"), ".akg", "tmp")
os.makedirs(_tmp_dir, exist_ok=True)
tempfile.tempdir = _tmp_dir

from akg_agents.op.agents.kernel_gen import KernelGen
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker, get_worker_manager
from akg_agents.core.async_pool.task_pool import TaskPool

sys.path.insert(0, str(Path(__file__).parent.parent))
from op.utils import (
    get_kernelbench_op_name, get_kernelbench_task_desc, add_op_prefix,
    generate_beautiful_test_report, get_device_id,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "python" / "akg_agents" / "op" / "config" / "cpp_coderonly_config.yaml"

# level1 每类选 2-3 个代表
# matmul(1,2), activation(19,21,23), norm(33,36), pool(42,45),
# reduce(47,49), conv(63), loss(94)
DEFAULT_INDICES = [1, 2, 19, 21, 23, 33, 36, 42, 45, 47, 49, 63, 94]


def py_compile_check(code: str) -> bool:
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix='.py', mode='w', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            tmp_path = f.name
        py_compile.compile(tmp_path, doraise=True)
        return True
    except py_compile.PyCompileError:
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def run_single_attempt(agent, verifier_worker, config, op_name, task_desc, task_id):
    """单次生成+验证，返回 (op_name, success, detail_dict)，兼容 generate_beautiful_test_report"""
    detail = {"gen_time_s": 0, "verify_time_s": 0, "error": ""}
    success = False

    # 1. KernelGen 生成
    t0 = time.time()
    try:
        generated_code, _, _ = await agent.run(
            op_name=op_name,
            task_desc=task_desc,
            dsl="cpp",
            framework="torch",
            backend="cpu",
            arch="x86_64",
            task_id=task_id,
        )
    except Exception as e:
        detail["error"] = f"生成失败: {e}"
        detail["gen_time_s"] = time.time() - t0
        return op_name, False, detail
    detail["gen_time_s"] = time.time() - t0

    # 2. py_compile
    if not py_compile_check(generated_code):
        detail["error"] = "py_compile 失败"
        return op_name, False, detail

    # 3. 代码结构
    if "class ModelNew" not in generated_code or "def forward" not in generated_code:
        detail["error"] = "缺少 ModelNew/forward"
        return op_name, False, detail

    # 4. KernelVerifier
    t1 = time.time()
    try:
        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=task_desc,
            task_id=task_id,
            framework="torch",
            dsl="cpp",
            backend="cpu",
            arch="x86_64",
            impl_func_name="ModelNew",
            config=config,
            worker=verifier_worker,
        )
        verify_result, error_log = await verifier.run(
            {"coder_code": generated_code}, device_id=0
        )
        success = bool(verify_result)
        if not success:
            detail["error"] = f"验证失败: {str(error_log)[:200]}"
    except Exception as e:
        detail["error"] = f"验证异常: {e}"
    detail["verify_time_s"] = time.time() - t1

    return op_name, success, detail


async def main():
    parser = argparse.ArgumentParser(description="KernelGen 评测 (pass@N, TaskPool 并发)")
    parser.add_argument(
        "--indices", type=str, default=None,
        help="逗号分隔的 task 序号，如 '19,21,23'。默认使用预选的代表性集合"
    )
    parser.add_argument(
        "--n", type=int, default=1,
        help="每个算子独立生成的次数，用于计算 pass@N（默认 1）"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="最大并发任务数（默认 4）"
    )
    parser.add_argument(
        "--tag", type=str, default="",
        help="结果文件的标签，如 'nojson' 或 'json'，用于对比"
    )
    args = parser.parse_args()

    n_attempts = max(1, args.n)

    if args.indices:
        indices = [int(x.strip()) for x in args.indices.split(",")]
    else:
        indices = DEFAULT_INDICES

    # 用已有工具函数获取任务
    benchmark_names = get_kernelbench_op_name(task_index_list=indices, framework="torch")
    if not benchmark_names:
        logger.error("未找到匹配的 KernelBench 文件")
        return

    logger.info(f"共 {len(benchmark_names)} 个算子, 每个跑 {n_attempts} 次, "
                f"总任务 {len(benchmark_names) * n_attempts}, 并发 {args.concurrency}")

    # 初始化
    agent = KernelGen()
    config = load_config(config_path=str(CONFIG_PATH))
    device_id = get_device_id()
    await register_local_worker([device_id], backend="cpu", arch="x86_64")
    worker = await get_worker_manager().select(backend="cpu", arch="x86_64")

    # 创建 TaskPool，提交所有任务
    pool = TaskPool(max_concurrency=args.concurrency)
    for bm_name in benchmark_names:
        task_desc = get_kernelbench_task_desc(bm_name, framework="torch")
        op_name = add_op_prefix(bm_name, benchmark="KernelBench")
        for attempt in range(1, n_attempts + 1):
            task_id = f"{op_name}_t{attempt}"
            pool.create_task(
                run_single_attempt,
                agent, worker, config, op_name, task_desc, task_id,
                task_name=task_id,
            )

    # 等待所有完成
    results = await pool.wait_all()

    # generate_beautiful_test_report 期望 [(op_name, result, _), ...]
    report_stats = generate_beautiful_test_report(
        results, config, "torch", "cpp", "cpu", "x86_64"
    )

    # 额外保存 JSON 详细结果
    tag_suffix = f"_{args.tag}" if args.tag else ""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"bench_kernel_gen{tag_suffix}_n{n_attempts}_{ts}.json")
    out_data = {
        "tag": args.tag,
        "n_attempts": n_attempts,
        "concurrency": args.concurrency,
        "timestamp": ts,
        "git_branch": os.popen("git rev-parse --abbrev-ref HEAD 2>/dev/null").read().strip(),
        "git_commit": os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
        "report_stats": report_stats,
        "results": [
            {"op_name": op, "success": suc, "detail": det}
            for op, suc, det in results
        ],
    }
    out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"结果已保存到: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
