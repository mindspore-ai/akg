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

"""代码执行与测试工具

已注册工具: test_with_reference
遗留函数: run_code, apply_patch（仅供 CLI 系统内部调用，v2 不注册）
v2 中 run_code 已由 core_v2 的 execute_script 替代，apply_patch 已由 edit_file 替代。
"""

import ast
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any

from akg_agents.op.tools.task_constructor.path_utils import resolve_path


def _ast_validate(task_code: str) -> str:
    """AST 结构检查，返回空字符串表示通过，否则返回错误描述"""
    try:
        tree = ast.parse(task_code)
    except SyntaxError as e:
        return f"语法错误: {e}"

    has_model = has_forward = has_get_inputs = has_get_init_inputs = False
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Model":
            has_model = True
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "forward":
                        has_forward = True
        elif isinstance(node, ast.FunctionDef):
            if node.name == "get_inputs":
                has_get_inputs = True
            elif node.name == "get_init_inputs":
                has_get_init_inputs = True

    issues = []
    if not has_model:
        issues.append("缺少 class Model(nn.Module)")
    if not has_forward:
        issues.append("缺少 Model.forward() 方法")
    if not has_get_inputs:
        issues.append("缺少 get_inputs() 函数")
    if not has_get_init_inputs:
        issues.append("缺少 get_init_inputs() 函数")
    return "; ".join(issues)


def test_with_reference(
    reference_code: str = "",
    task_file: str = "",
    task_code: str = "",
    multi_inputs_code: str = "",
    timeout: int = 120,
    workspace_dir: str = "",
    output_dir: str = "",
) -> Dict[str, Any]:
    """
    统一验证+对比测试工具。

    始终执行:
      1) AST 结构检查 (Model, forward, get_inputs, get_init_inputs)
      2) 运行时验证 (实例化, forward, NaN/Inf 检查)
    当 reference_code 非空时额外执行:
      3) 与 reference_forward 对比，支持 multi_inputs_code 多组输入
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

    # ---- Phase 1: AST 结构检查 ----
    ast_err = _ast_validate(task_code)
    if ast_err:
        return {"status": "error", "output": "",
                "error_information": f"格式问题: {ast_err}"}

    has_reference = bool(reference_code and reference_code.strip())

    # 保存测试文件到输出目录
    if has_reference and od and od.is_dir():
        try:
            (od / "reference_code.py").write_text(reference_code, encoding="utf-8")
            if multi_inputs_code.strip():
                (od / "multi_inputs_code.py").write_text(multi_inputs_code, encoding="utf-8")
        except Exception:
            pass

    # ---- Phase 2 & 3: 运行时验证 + 可选对比测试 ----
    test_script = task_code + "\n\n"

    if has_reference:
        test_script += "# === Reference ===\n" + reference_code + "\n\n"
    if multi_inputs_code.strip():
        test_script += multi_inputs_code + "\n\n"

    # 运行时验证 + 对比测试的统一脚本
    test_script += (
        "import torch\n"
        "import traceback\n\n"
    )

    if has_reference:
        test_script += (
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
        )

    test_script += (
        "try:\n"
        "    init_inputs = get_init_inputs()\n"
        "    model = Model(*init_inputs)\n"
        "    inputs = get_inputs()\n"
        "    output = model.forward(*inputs)\n"
        "    # --- 运行时验证 ---\n"
        "    def _check_tensor(t, prefix=''):\n"
        "        if isinstance(t, torch.Tensor):\n"
        "            has_nan = torch.isnan(t).any().item()\n"
        "            has_inf = torch.isinf(t).any().item()\n"
        "            print(f'{prefix}shape={t.shape}, dtype={t.dtype}, nan={has_nan}, inf={has_inf}')\n"
        "            if has_nan: print(f'WARNING: {prefix}output contains NaN')\n"
        "            if has_inf: print(f'WARNING: {prefix}output contains Inf')\n"
        "    if isinstance(output, torch.Tensor):\n"
        "        _check_tensor(output)\n"
        "    elif isinstance(output, (tuple, list)):\n"
        "        for i, t in enumerate(output):\n"
        "            _check_tensor(t, f'output[{i}]: ')\n"
        "    print('VALIDATION_OK')\n"
    )

    if has_reference:
        test_script += (
            "    # --- 对比测试 ---\n"
            "    all_pass = True\n"
            "    test_cases = []\n"
            "    if 'get_multi_test_inputs' in dir():\n"
            "        test_cases = get_multi_test_inputs()\n"
            "    else:\n"
            "        test_cases = [{'name': 'default', 'inputs': get_inputs()}]\n"
            "    for case in test_cases:\n"
            "        name = case.get('name', 'unnamed')\n"
            "        c_inputs = case['inputs']\n"
            "        case_init = case.get('init_inputs', init_inputs)\n"
            "        if case_init is not init_inputs:\n"
            "            model = Model(*case_init)\n"
            "        print(f'--- Case: {name} ---')\n"
            "        try:\n"
            "            model_out = model.forward(*c_inputs)\n"
            "            ref_out = reference_forward(c_inputs, case_init)\n"
            "            if not _compare(name, model_out, ref_out):\n"
            "                all_pass = False\n"
            "        except Exception as e:\n"
            "            print(f'  [ERROR] {name}: {e}')\n"
            "            all_pass = False\n"
            "    if all_pass:\n"
            "        print('\\nALL_TESTS_PASSED')\n"
            "    else:\n"
            "        print('\\nSOME_TESTS_FAILED')\n"
        )

    test_script += (
        "except Exception as e:\n"
        "    print(f'VALIDATION_ERROR: {type(e).__name__}: {e}')\n"
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

        # 解析结果：优先检查对比测试结果，其次检查验证结果
        if has_reference and "ALL_TESTS_PASSED" in stdout:
            return {"status": "success",
                    "output": f"验证通过 + 所有对比测试通过\n{stdout}",
                    "error_information": ""}
        elif has_reference and "SOME_TESTS_FAILED" in stdout:
            return {"status": "error",
                    "output": stdout,
                    "error_information": "部分对比测试失败，请检查输出"}
        elif "VALIDATION_OK" in stdout and not has_reference:
            warn_parts = []
            if "WARNING:" in stdout and "NaN" in stdout:
                warn_parts.append("输出包含 NaN")
            if "WARNING:" in stdout and "Inf" in stdout:
                warn_parts.append("输出包含 Inf")
            output_msg = f"验证通过\n{stdout}"
            if warn_parts:
                output_msg += f"\n警告: {'; '.join(warn_parts)}"
            return {"status": "success", "output": output_msg, "error_information": ""}
        elif "VALIDATION_ERROR" in stdout:
            error_line = [l for l in stdout.splitlines() if "VALIDATION_ERROR" in l]
            return {"status": "error", "output": stdout,
                    "error_information": error_line[0] if error_line else "运行时验证失败"}
        else:
            combined = stdout + ("\n" + stderr if stderr else "")
            return {"status": "error", "output": combined,
                    "error_information": f"执行错误 (exit code {result.returncode})"}
    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "",
                "error_information": f"超时 ({timeout}s)"}
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
