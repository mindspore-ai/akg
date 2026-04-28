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
KernelGen 去 JSON 化测试

验证 KernelGen 直接输出纯 Python 代码（非 JSON），并通过：
1. py_compile 语法校验
2. KernelVerifier 正确性验证

运行方式：
    cd akg/aikg && source env.sh
    python tests/st/test_kernel_gen_dejson.py
"""

import asyncio
import logging
import os
import sys
import textwrap
import py_compile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from akg_agents.op.agents.kernel_gen import KernelGen
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker, get_worker_manager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s'
)
logger = logging.getLogger(__name__)

os.environ['AKG_AGENTS_STREAM_OUTPUT'] = 'on'

_tmp_dir = os.path.join(os.path.expanduser("~"), ".akg", "tmp")
os.makedirs(_tmp_dir, exist_ok=True)
tempfile.tempdir = _tmp_dir

RELU_TASK_DESC = textwrap.dedent("""\
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
    x = torch.randn(batch_size, dim).to(torch.bfloat16)
    return [x]


def get_init_inputs():
    return []  # No special initialization inputs needed
""")


def py_compile_check(code: str) -> bool:
    """使用 py_compile 验证代码语法"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix='.py', mode='w', delete=False, encoding='utf-8'
        ) as f:
            f.write(code)
            tmp_path = f.name
        py_compile.compile(tmp_path, doraise=True)
        return True
    except py_compile.PyCompileError as e:
        logger.error(f"py_compile 失败: {e}")
        return False
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def test_kernel_gen_relu_cpu():
    """端到端测试：KernelGen 生成 ReLU CPU 代码 -> py_compile -> KernelVerifier 验证"""

    logger.info("=" * 60)
    logger.info("测试: KernelGen 去 JSON 化 - ReLU CPU")
    logger.info("=" * 60)

    # ==================== 步骤 1: KernelGen 生成代码 ====================
    logger.info("步骤 1: 创建 KernelGen 并生成 ReLU 代码...")
    agent = KernelGen()

    generated_code, formatted_prompt, reasoning = await agent.run(
        op_name="relu",
        task_desc=RELU_TASK_DESC,
        dsl="cpp",
        framework="torch",
        backend="cpu",
        arch="x86_64",
        task_id="test_dejson_relu",
    )

    logger.info(f"生成代码长度: {len(generated_code)}")
    logger.info(f"生成代码前 200 字符:\n{generated_code[:200]}")

    # ==================== 步骤 2: 验证非 JSON 格式 ====================
    logger.info("步骤 2: 验证输出不是 JSON 格式...")
    stripped = generated_code.strip()
    assert not stripped.startswith('{'), "生成的代码不应该是 JSON 格式（以 '{' 开头）"
    assert not stripped.startswith('{"'), "生成的代码不应该是 JSON 格式（以 '{\"' 开头）"
    logger.info("通过: 输出不是 JSON 格式")

    # ==================== 步骤 3: py_compile 语法校验 ====================
    logger.info("步骤 3: py_compile 语法校验...")
    assert py_compile_check(generated_code), "py_compile 语法校验失败"
    logger.info("通过: py_compile 语法校验成功")

    # ==================== 步骤 4: 验证代码包含 ModelNew ====================
    logger.info("步骤 4: 验证代码结构...")
    assert "class ModelNew" in generated_code, "代码应包含 class ModelNew"
    assert "def forward" in generated_code, "代码应包含 def forward 方法"
    logger.info("通过: 代码结构验证成功")

    # ==================== 步骤 5: KernelVerifier 正确性验证 ====================
    logger.info("步骤 5: KernelVerifier 正确性验证...")

    device_id = int(os.environ.get("DEVICE_ID", "0"))
    config_path = Path(__file__).parent.parent.parent / "python" / "akg_agents" / "op" / "config" / "cpp_coderonly_config.yaml"
    config = load_config(config_path=str(config_path))

    await register_local_worker([device_id], backend="cpu", arch="x86_64")
    worker = await get_worker_manager().select(backend="cpu", arch="x86_64")
    if not worker:
        raise RuntimeError("无法获取 CPU worker，请确认 register_local_worker 已成功")

    verifier = KernelVerifier(
        op_name="relu",
        framework_code=RELU_TASK_DESC,
        task_id="test_dejson_relu",
        framework="torch",
        dsl="cpp",
        backend="cpu",
        arch="x86_64",
        impl_func_name="ModelNew",
        config=config,
        worker=worker,
    )

    task_info = {"coder_code": generated_code}
    result, error_log = await verifier.run(task_info, device_id=device_id)
    assert result, f"KernelVerifier 验证失败: {error_log}"
    logger.info("通过: KernelVerifier 正确性验证成功")

    # ==================== 总结 ====================
    logger.info("=" * 60)
    logger.info("所有验证通过!")
    logger.info("  - 非 JSON 格式输出")
    logger.info("  - py_compile 语法校验通过")
    logger.info("  - 代码结构正确 (ModelNew + forward)")
    logger.info("  - KernelVerifier 正确性验证通过")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("生成的完整代码:")
    print("=" * 60)
    print(generated_code)
    print("=" * 60)

    return True


async def main():
    try:
        success = await test_kernel_gen_relu_cpu()
        if success:
            logger.info("测试成功!")
            sys.exit(0)
        else:
            logger.error("测试失败!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"测试异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
