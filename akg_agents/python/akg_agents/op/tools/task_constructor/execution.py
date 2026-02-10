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

"""代码执行与测试工具：run_code、test_with_reference、apply_patch"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

from akg_agents.op.tools.task_constructor.path_utils import resolve_path


def test_with_reference(
    reference_code: str,
    task_file: str = "",
    task_code: str = "",
    multi_inputs_code: str = "",
    timeout: int = 120,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """
    对比测试：将生成的 Model 与 reference 函数对比，支持多组输入。
    reference_code 必须定义 reference_forward(inputs, init_inputs) 函数。
    """
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None

    if task_file and not task_code:
        path = resolve_path(task_file, workspace_dir=ws, output_dir=od)
        if path.exists():
            task_code = path.read_text(encoding="utf-8")
        else:
            return {"status": "error", "output": "",
                    "error_information": f"任务文件不存在: {path}"}

    if not task_code.strip():
        return {"status": "error", "output": "", "error_information": "task_code 不能为空"}
    if not reference_code.strip():
        return {"status": "error", "output": "", "error_information": "reference_code 不能为空"}

    # 保存测试文件到输出目录，方便复现
    if od and od.is_dir():
        try:
            (od / "reference_code.py").write_text(reference_code, encoding="utf-8")
            if multi_inputs_code.strip():
                (od / "multi_inputs_code.py").write_text(multi_inputs_code, encoding="utf-8")
        except Exception:
            pass
    elif ws and ws.is_dir():
        try:
            (ws / "reference_code.py").write_text(reference_code, encoding="utf-8")
            if multi_inputs_code.strip():
                (ws / "multi_inputs_code.py").write_text(multi_inputs_code, encoding="utf-8")
        except Exception:
            pass

    # 构建对比测试脚本
    test_script = (
        task_code + "\n\n"
        "# === Reference ===\n"
        + reference_code + "\n\n"
    )

    if multi_inputs_code.strip():
        test_script += multi_inputs_code + "\n\n"

    test_script += (
        "import torch\n"
        "import traceback\n\n"
        "def _compare(name, model_out, ref_out, rtol=1e-3, atol=1e-3):\n"
        "    if isinstance(model_out, torch.Tensor) and isinstance(ref_out, torch.Tensor):\n"
        "        try:\n"
        "            match = torch.allclose(model_out.float(), ref_out.float(), rtol=rtol, atol=atol)\n"
        "        except Exception:\n"
        "            match = False\n"
        "        status = 'PASS' if match else 'FAIL'\n"
        "        max_diff = (model_out.float() - ref_out.float()).abs().max().item() if model_out.shape == ref_out.shape else float('inf')\n"
        "        print(f'  [{status}] {name}: shape={model_out.shape}, max_diff={max_diff:.6e}')\n"
        "        return match\n"
        "    elif isinstance(model_out, (tuple, list)) and isinstance(ref_out, (tuple, list)):\n"
        "        all_match = True\n"
        "        for i, (m, r) in enumerate(zip(model_out, ref_out)):\n"
        "            if not _compare(f'{name}[{i}]', m, r, rtol, atol):\n"
        "                all_match = False\n"
        "        return all_match\n"
        "    else:\n"
        "        match = str(model_out) == str(ref_out)\n"
        "        print(f'  [{\"PASS\" if match else \"FAIL\"}] {name}: type mismatch')\n"
        "        return match\n\n"
        "try:\n"
        "    init_inputs = get_init_inputs()\n"
        "    model = Model(*init_inputs)\n"
        "    all_pass = True\n"
        "    test_cases = []\n\n"
        "    if 'get_multi_test_inputs' in dir():\n"
        "        test_cases = get_multi_test_inputs()\n"
        "    else:\n"
        "        test_cases = [{'name': 'default', 'inputs': get_inputs()}]\n\n"
        "    for case in test_cases:\n"
        "        name = case.get('name', 'unnamed')\n"
        "        inputs = case['inputs']\n"
        "        case_init = case.get('init_inputs', init_inputs)\n"
        "        if case_init is not init_inputs:\n"
        "            model = Model(*case_init)\n"
        "        print(f'--- Case: {name} ---')\n"
        "        try:\n"
        "            model_out = model.forward(*inputs)\n"
        "            ref_out = reference_forward(inputs, case_init)\n"
        "            if not _compare(name, model_out, ref_out):\n"
        "                all_pass = False\n"
        "        except Exception as e:\n"
        "            print(f'  [ERROR] {name}: {e}')\n"
        "            all_pass = False\n\n"
        "    if all_pass:\n"
        "        print('\\nALL_TESTS_PASSED')\n"
        "    else:\n"
        "        print('\\nSOME_TESTS_FAILED')\n"
        "except Exception as e:\n"
        "    print(f'TEST_ERROR: {e}')\n"
        "    traceback.print_exc()\n"
    )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8") as f:
            f.write(test_script)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if "ALL_TESTS_PASSED" in stdout:
            return {"status": "success",
                    "output": f"所有测试通过\n{stdout}",
                    "error_information": ""}
        elif "SOME_TESTS_FAILED" in stdout:
            return {"status": "error",
                    "output": stdout,
                    "error_information": "部分测试失败，请检查输出"}
        else:
            combined = stdout + ("\n" + stderr if stderr else "")
            return {"status": "error", "output": combined,
                    "error_information": f"测试执行错误 (exit code {result.returncode})"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "",
                "error_information": f"测试超时 ({timeout}s)"}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def run_code(
    code: str = "",
    file_path: str = "",
    timeout: int = 60,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """运行 Python 代码"""
    tmp_path = None
    try:
        cwd = None
        if output_dir and Path(output_dir).is_dir():
            cwd = str(Path(output_dir).resolve())
        elif workspace_dir and Path(workspace_dir).is_dir():
            cwd = str(Path(workspace_dir).resolve())

        if file_path:
            ws = Path(workspace_dir) if workspace_dir else None
            od = Path(output_dir) if output_dir else None
            path = resolve_path(file_path, workspace_dir=ws, output_dir=od)
            if not path.exists():
                return {"status": "error", "output": "",
                        "error_information": f"文件不存在: {path}"}
            cmd = [sys.executable, str(path)]
        elif code:
            tmp = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False, encoding="utf-8")
            tmp.write(code)
            tmp.close()
            tmp_path = tmp.name
            cmd = [sys.executable, tmp_path]
        else:
            return {"status": "error", "output": "", "error_information": "需要 code 或 file_path"}

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        combined = ""
        if stdout:
            combined += f"[stdout]\n{stdout}\n"
        if stderr:
            combined += f"[stderr]\n{stderr}\n"

        if result.returncode == 0:
            return {"status": "success", "output": combined or "success, no output",
                    "error_information": ""}
        else:
            return {"status": "error", "output": combined,
                    "error_information": f"exit code {result.returncode}"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "", "error_information": f"timeout ({timeout}s)"}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


def apply_patch(
    file_path: str,
    old_string: str,
    new_string: str,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """通过 old_string/new_string 修改文件"""
    ws = Path(workspace_dir) if workspace_dir else None
    od = Path(output_dir) if output_dir else None
    path = resolve_path(file_path, workspace_dir=ws, output_dir=od)

    if old_string == "":
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_string, encoding="utf-8")
            return {"status": "success", "output": f"文件已创建/写入: {path}",
                    "error_information": ""}
        except Exception as e:
            return {"status": "error", "output": "", "error_information": str(e)}

    if not path.exists():
        return {"status": "error", "output": "",
                "error_information": f"文件不存在: {path}"}

    try:
        content = path.read_text(encoding="utf-8")
        if old_string not in content:
            lines = content.split("\n")
            old_lines = old_string.split("\n")
            found = False
            for i in range(len(lines) - len(old_lines) + 1):
                block = lines[i:i + len(old_lines)]
                if all(a.strip() == b.strip() for a, b in zip(block, old_lines)):
                    original_block = "\n".join(block)
                    content = content.replace(original_block, new_string, 1)
                    found = True
                    break
            if not found:
                return {"status": "error", "output": "",
                        "error_information": "old_string 在文件中未找到"}
        else:
            count = content.count(old_string)
            if count > 1:
                return {"status": "error", "output": "",
                        "error_information": f"old_string 有 {count} 处匹配"}
            content = content.replace(old_string, new_string, 1)

        path.write_text(content, encoding="utf-8")
        return {"status": "success", "output": f"已修改: {path}", "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}
