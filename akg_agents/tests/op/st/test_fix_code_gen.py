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
FixCodeGen ST 测试（真实 LLM 调用）

场景：一段缺少 import torch 的 Python kernel 代码，喂给 FixCodeGen，
验证修复后的代码能通过 py_compile 且包含 import torch。

运行方式：
    cd akg/akg_agents && source env.sh
    python tests/op/st/test_fix_code_gen.py
"""

import asyncio
import logging
import os
import py_compile
import sys
import tempfile
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from akg_agents.core_v2.agents import AgentBase
from akg_agents.op.utils.diff_utils import DiffApplier, parse_modifications

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s() - %(message)s",
)
logger = logging.getLogger(__name__)

os.environ["AKG_AGENTS_STREAM_OUTPUT"] = "on"

_tmp_dir = os.path.join(os.path.expanduser("~"), ".akg", "tmp")
os.makedirs(_tmp_dir, exist_ok=True)
tempfile.tempdir = _tmp_dir

# 缺少 import torch / import torch.nn as nn 的代码
BROKEN_CODE = textwrap.dedent("""\
class ModelNew(nn.Module):

    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


batch_size = 16
dim = 16384


def get_inputs():
    x = torch.randn(batch_size, dim).to(torch.bfloat16)
    return [x]


def get_init_inputs():
    return []
""")

ERROR_LOG = textwrap.dedent("""\
Traceback (most recent call last):
  File "generated_code.py", line 1, in <module>
    class ModelNew(nn.Module):
                   ^^
NameError: name 'nn' is not defined
""")

CONDUCTOR_SUGGESTION = (
    "代码缺少 import torch 和 import torch.nn as nn，"
    "请在文件顶部添加相应的 import 语句。"
)


def py_compile_check(code: str) -> bool:
    """使用 py_compile 验证代码语法"""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".py", mode="w", delete=False, encoding="utf-8"
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


async def test_fix_code_gen_missing_import():
    """ST 测试：FixCodeGen 修复缺少 import torch 的代码"""

    logger.info("=" * 60)
    logger.info("ST 测试: FixCodeGen 修复缺少 import torch 的代码")
    logger.info("=" * 60)

    # === 步骤 1: 确认原始代码有语法/运行问题 ===
    logger.info("步骤 1: 确认原始代码不包含 import torch ...")
    assert "import torch" not in BROKEN_CODE, "测试前提失败: 原始代码不应包含 import torch"
    logger.info("✓ 原始代码确实缺少 import torch")

    # === 步骤 2: 调用 LLM 生成修复方案 ===
    logger.info("步骤 2: 创建 AgentBase 并调用 LLM 生成修复方案 ...")

    context = {
        "agent_name": "fix_code_gen",
        "session_id": "",
        "task_id": "st_test_fix_import",
        "op_name": "relu",
        "dsl": "triton_cuda",
        "backend": "cuda",
        "arch": "a100",
        "framework": "torch",
        "workflow_name": "coder_only",
        "task_desc": "ReLU operator",
        "hash": "",
    }
    config = {
        "agent_model_config": {"fix_code_gen": "fast", "default": "fast"},
    }

    agent_base = AgentBase(context=context, config=config)
    prompt_template = agent_base.load_template("fix_code_gen/edit.j2")

    input_data = {
        "dsl": "triton_cuda",
        "expert_suggestion": "",
        "op_name": "relu",
        "framework": "torch",
        "task_desc": "ReLU operator",
        "original_code": BROKEN_CODE,
        "error_log": ERROR_LOG,
        "conductor_suggestion": CONDUCTOR_SUGGESTION,
    }

    response_text, prompt, reasoning = await agent_base.run_llm(
        prompt=prompt_template,
        input=input_data,
        model_level="fast",
    )

    logger.info(f"LLM 返回长度: {len(response_text)}")
    logger.info(f"LLM 返回内容前 300 字符:\n{response_text[:300]}")

    # === 步骤 3: 解析修改方案 ===
    logger.info("步骤 3: 解析 LLM 返回的修改方案 ...")
    modifications = parse_modifications(response_text)
    assert len(modifications) > 0, (
        f"未解析到任何修改指令。LLM 返回: {response_text[:500]}"
    )
    logger.info(f"✓ 解析到 {len(modifications)} 个修改指令")
    for i, mod in enumerate(modifications):
        logger.info(f"  修改 {i + 1}: reason={mod.reason}")

    # === 步骤 4: 应用修改 ===
    logger.info("步骤 4: 应用修改 ...")
    result = DiffApplier.apply_modifications(BROKEN_CODE, modifications)

    logger.info(f"应用结果: success={result.success}, applied={result.applied_count}")
    if result.errors:
        for err in result.errors:
            logger.warning(f"  应用错误: {err}")
    if result.diff_text:
        logger.info(f"Diff:\n{result.diff_text}")

    assert result.success, (
        f"修改应用失败。errors: {result.errors}"
    )
    assert result.applied_count >= 1, "至少应有 1 个修改成功应用"
    logger.info(f"✓ 成功应用 {result.applied_count} 处修改")

    # === 步骤 5: 验证修复结果 ===
    fixed_code = result.modified_code

    logger.info("步骤 5a: py_compile 语法校验 ...")
    assert py_compile_check(fixed_code), (
        f"py_compile 语法校验失败。修复后的代码:\n{fixed_code}"
    )
    logger.info("✓ py_compile 语法校验通过")

    logger.info("步骤 5b: 验证包含 import torch ...")
    assert "import torch" in fixed_code, (
        f"修复后的代码不包含 import torch。代码:\n{fixed_code}"
    )
    logger.info("✓ 修复后的代码包含 import torch")

    logger.info("步骤 5c: 验证原有代码结构未被破坏 ...")
    assert "class ModelNew" in fixed_code, "修复后的代码应保留 class ModelNew"
    assert "def forward" in fixed_code, "修复后的代码应保留 def forward"
    assert "torch.relu" in fixed_code, "修复后的代码应保留 torch.relu 调用"
    logger.info("✓ 原有代码结构完整保留")

    # === 总结 ===
    logger.info("=" * 60)
    logger.info("所有验证通过!")
    logger.info("  - LLM 生成了有效的修改方案")
    logger.info(f"  - 成功应用 {result.applied_count} 处修改")
    logger.info("  - py_compile 语法校验通过")
    logger.info("  - import torch 已添加")
    logger.info("  - 原有代码结构未被破坏")
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("修复后的完整代码:")
    print("=" * 60)
    print(fixed_code)
    print("=" * 60)

    return True


async def main():
    try:
        success = await test_fix_code_gen_missing_import()
        if success:
            logger.info("ST 测试成功!")
            sys.exit(0)
        else:
            logger.error("ST 测试失败!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"ST 测试异常: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
