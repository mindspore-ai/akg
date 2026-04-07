#!/usr/bin/env python3
# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
Triton-CUDA 参考数据批量生成脚本

在 CUDA 环境中运行 SGLang / vLLM 的 triton_cuda 算子，批量生成 .pt 参考数据缓存。
生成的 .pt 文件包含 inputs + outputs + init_inputs，可 scp 到 Ascend 环境后
直接用于 triton_ascend 算子生成和验证（无需 CUDA 运行时）。

前置条件：
  - CUDA GPU 可用（torch.cuda.is_available()）
  - source env.sh
  - KernelBench 子模块已初始化（部分 benchmark 文件在 thirdparty/ 下）

运行方式：
  # 生成全部（sglang + vllm triton_ops + vllm torch_ops）
  python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py

  # 只生成 sglang
  python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py --source sglang

  # 只生成 vllm triton_ops
  python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py --source vllm_triton

  # 只生成 vllm torch_ops
  python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py --source vllm_torch

  # 指定算子
  python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py --source sglang --ops triton_tanh merge_state_triton

  # 指定输出目录和设备
  python reproduce/wip/triton-cuda-to-ascend/gen_reference_cache.py --output-dir ./my_cache --device 1

产出：
  <output_dir>/
  ├── manifest.json           # 汇总清单
  ├── sglang/
  │   ├── triton_tanh.pt
  │   ├── merge_state_triton.pt
  │   └── ...
  ├── vllm_triton/
  │   ├── rms_norm_kernel.pt
  │   └── ...
  └── vllm_torch/
      ├── silu_and_mul.pt
      └── ...
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("gen_reference_cache")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

BENCHMARK_BASE = PROJECT_ROOT / "benchmark" / "akg_kernels_bench" / "thirdparty"
DEFAULT_OUTPUT_DIR = os.path.expanduser("~/.akg/.tmp/reference_data/triton_cuda_cache")

SOURCES = {
    "sglang": {
        "path": BENCHMARK_BASE / "sglang",
        "dsl": "triton_cuda",
        "exclude_dirs": ["class_method"],
    },
    "vllm_triton": {
        "path": BENCHMARK_BASE / "vllm" / "triton_ops",
        "dsl": "triton_cuda",
        "exclude_dirs": [],
    },
    "vllm_torch": {
        "path": BENCHMARK_BASE / "vllm" / "torch_ops",
        "dsl": "triton_cuda",
        "exclude_dirs": [],
    },
}


def discover_ops(source_name: str, ops_filter: list = None) -> list:
    """发现指定 source 下的所有算子文件，返回 [(op_name, file_path), ...]"""
    source_cfg = SOURCES[source_name]
    base_path = source_cfg["path"]
    exclude_dirs = source_cfg["exclude_dirs"]

    if not base_path.exists():
        logger.warning(f"路径不存在: {base_path}")
        return []

    results = []
    for f in sorted(base_path.iterdir()):
        if f.is_dir() and f.name in exclude_dirs:
            continue
        if f.is_dir():
            continue
        if not f.suffix == ".py" or f.name.startswith("__"):
            continue
        op_name = f.stem
        if ops_filter and op_name not in ops_filter:
            continue
        results.append((op_name, str(f)))
    return results


async def gen_reference_for_op(
    op_name: str,
    op_file: str,
    source_name: str,
    output_dir: str,
    device_id: int,
    timeout: int,
) -> dict:
    """为单个算子生成参考数据"""
    from akg_agents.op.verifier.kernel_verifier import KernelVerifier
    from akg_agents.op.config.config_validator import load_config
    from akg_agents.core.worker.manager import get_worker_manager

    source_cfg = SOURCES[source_name]
    dsl = source_cfg["dsl"]
    framework = "torch"
    backend = "cuda"
    arch = "a100"

    result = {
        "op_name": op_name,
        "source": source_name,
        "success": False,
        "pt_path": None,
        "error": None,
        "elapsed_s": 0,
    }
    t0 = time.time()

    try:
        with open(op_file, "r", encoding="utf-8") as f:
            framework_code = f.read()

        config = load_config(dsl, backend=backend)

        worker = await get_worker_manager().select(backend=backend, arch=arch)
        if not worker:
            result["error"] = f"No worker available for {backend}/{arch}"
            return result

        verifier = KernelVerifier(
            op_name=op_name,
            framework_code=framework_code,
            task_id=f"ref_cache_{source_name}_{op_name}",
            framework=framework,
            dsl=dsl,
            backend=backend,
            arch=arch,
            config=config,
            worker=worker,
        )

        success, log, ref_bytes = await verifier.generate_reference_data(
            framework_code, save_inputs=True, timeout=timeout
        )

        if not success:
            result["error"] = log[:500]
            return result

        if len(ref_bytes) == 0:
            result["error"] = "Empty reference data"
            return result

        sub_dir = os.path.join(output_dir, source_name)
        os.makedirs(sub_dir, exist_ok=True)
        pt_path = os.path.join(sub_dir, f"{op_name}.pt")
        with open(pt_path, "wb") as f:
            f.write(ref_bytes)

        result["success"] = True
        result["pt_path"] = pt_path
        result["size_bytes"] = len(ref_bytes)

    except Exception as e:
        result["error"] = str(e)[:500]
    finally:
        result["elapsed_s"] = round(time.time() - t0, 1)

    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量生成 SGLang/vLLM triton_cuda 参考数据缓存",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source", nargs="+", default=list(SOURCES.keys()),
        choices=list(SOURCES.keys()),
        help="指定数据源（默认全部）",
    )
    parser.add_argument(
        "--ops", nargs="+", default=None,
        help="只生成指定算子（默认全部）",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"输出目录（默认 {DEFAULT_OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--device", type=int, default=int(os.getenv("DEVICE_ID", "0")),
        help="CUDA 设备 ID（默认 $DEVICE_ID 或 0）",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="单个算子超时时间/秒（默认 120）",
    )
    parser.add_argument(
        "--concurrency", type=int, default=1,
        help="并行度（默认 1，参考数据生成建议串行以避免 GPU OOM）",
    )
    return parser.parse_args()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args()
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    from akg_agents.core.worker.manager import register_local_worker
    await register_local_worker([args.device], backend="cuda", arch="a100")

    all_ops = []
    for source_name in args.source:
        ops = discover_ops(source_name, args.ops)
        logger.info(f"[{source_name}] 发现 {len(ops)} 个算子")
        for op_name, op_file in ops:
            all_ops.append((source_name, op_name, op_file))

    if not all_ops:
        logger.error("未发现任何算子，请检查 --source 和 --ops 参数")
        return

    print(f"\n{'='*70}")
    print(f"  Triton-CUDA 参考数据批量生成")
    print(f"{'='*70}")
    print(f"  算子总数:   {len(all_ops)}")
    print(f"  数据源:     {', '.join(args.source)}")
    print(f"  输出目录:   {output_dir}")
    print(f"  CUDA 设备:  {args.device}")
    print(f"  超时/算子:  {args.timeout}s")
    print(f"{'='*70}\n")

    results = []
    success_count = 0
    fail_count = 0

    for i, (source_name, op_name, op_file) in enumerate(all_ops):
        print(f"[{i+1}/{len(all_ops)}] {source_name}/{op_name} ... ", end="", flush=True)

        r = await gen_reference_for_op(
            op_name, op_file, source_name, output_dir, args.device, args.timeout
        )
        results.append(r)

        if r["success"]:
            success_count += 1
            size_kb = r.get("size_bytes", 0) / 1024
            print(f"OK ({r['elapsed_s']}s, {size_kb:.1f}KB)")
        else:
            fail_count += 1
            print(f"FAIL ({r['elapsed_s']}s) - {r['error'][:80]}")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": output_dir,
        "device": args.device,
        "total": len(results),
        "success": success_count,
        "failed": fail_count,
        "ops": results,
    }

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"  完成")
    print(f"{'='*70}")
    print(f"  成功: {success_count}/{len(results)}")
    print(f"  失败: {fail_count}/{len(results)}")
    print(f"  清单: {manifest_path}")
    print(f"  输出: {output_dir}")
    print(f"{'='*70}")

    if fail_count > 0:
        print(f"\n  失败算子:")
        for r in results:
            if not r["success"]:
                print(f"    - {r['source']}/{r['op_name']}: {r['error'][:100]}")

    print(f"\n  下一步：将 {output_dir} 拷贝到 Ascend 环境后运行：")
    print(f"  python reproduce/wip/triton-cuda-to-ascend/run_adaptive_search_with_cache.py \\")
    print(f"      --source sglang")


if __name__ == "__main__":
    asyncio.run(main())
