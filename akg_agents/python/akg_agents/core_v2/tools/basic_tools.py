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
通用基础工具集 (core_v2)

提供与领域无关的通用文件操作、代码执行、交互工具。
所有工具函数只接受普通路径参数，不依赖 workspace/output 等子 agent 概念。
路径的预解析由调用方（如 TaskConstructor）负责。
"""

import re
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

READ_FILE_MAX_LINES = 300
MAX_SCRIPT_OUTPUT_CHARS = 30000


# ==================== 内部辅助 ====================


def _is_binary_file(path: Path, sample_size: int = 8192) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
        if b"\x00" in chunk:
            return True
        non_text = sum(1 for b in chunk if b < 8 or (b > 13 and b < 32))
        return non_text / max(len(chunk), 1) > 0.3
    except Exception:
        return False


def _truncate_script_output(text: str, max_chars: int = MAX_SCRIPT_OUTPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [输出已截断: 总计 {len(text)} 字符] ...\n\n"
        + text[-half:]
    )


# ==================== 文件操作工具 ====================


def read_file(
    file_path: str,
    offset: int = None,
    limit: int = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """读取文件内容（带行号）。

    超长文件自动截断到 READ_FILE_MAX_LINES 行。
    二进制文件返回文件元信息而不是内容。
    """
    logger.info(f"[read_file] 读取文件: {file_path}")

    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if not path.exists():
        return {"status": "error", "output": "",
                "error_information": f"文件不存在: {path}"}
    if not path.is_file():
        if path.is_dir():
            return {"status": "error", "output": "",
                    "error_information": f"这是目录: {path}\n请使用 scan_dir 查看。"}
        return {"status": "error", "output": "", "error_information": f"不是文件: {path}"}

    if _is_binary_file(path):
        size = path.stat().st_size
        return {
            "status": "success",
            "output": f"[二进制文件: {path.name}, 大小: {size} bytes, 后缀: {path.suffix}]\n无法显示二进制内容。",
            "error_information": ""
        }

    try:
        all_lines = path.read_text(encoding=encoding).splitlines(keepends=True)
        total = len(all_lines)

        if offset is not None:
            start = max(int(offset) - 1, 0)
            end = start + int(limit) if limit else min(start + READ_FILE_MAX_LINES, total)
            lines = all_lines[start:end]
            line_offset = start + 1
        else:
            if total > READ_FILE_MAX_LINES and limit is None:
                lines = all_lines[:READ_FILE_MAX_LINES]
                line_offset = 1
            else:
                lines = all_lines
                line_offset = 1

        numbered = [f"{line_offset + i:>5}| {line.rstrip()}" for i, line in enumerate(lines)]
        content = "\n".join(numbered)

        meta = f"[文件: {path.name}, 总行数: {total}]"
        if offset is not None:
            end_line = line_offset + len(lines) - 1
            meta += f" [显示: 第{line_offset}-{end_line}行]"
        elif total > READ_FILE_MAX_LINES and limit is None:
            meta += (f" [已截断: 仅显示前{READ_FILE_MAX_LINES}行]"
                     f" 提示: 使用 offset/limit 参数查看后续内容")

        output = f"{meta}\n{content}"
        return {"status": "success", "output": output, "error_information": ""}
    except UnicodeDecodeError as e:
        return {"status": "error", "output": "",
                "error_information": f"文件编码错误 ({encoding}): {str(e)}。尝试使用其他 encoding 参数。"}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def write_file(
    file_path: str,
    content: str,
    overwrite: bool = True,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """写入文件。

    自动创建父目录。覆盖已有文件时自动创建 .bak 备份。
    """
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = path.resolve()

    if path.exists() and not overwrite:
        return {"status": "error", "output": "",
                "error_information": f"文件已存在: {path}。设置 overwrite=true 以覆盖。"}

    try:
        path.parent.mkdir(parents=True, exist_ok=True)

        backup_info = ""
        if path.exists() and overwrite:
            backup_path = path.with_suffix(path.suffix + ".bak")
            try:
                import shutil
                shutil.copy2(path, backup_path)
                backup_info = f" (备份: {backup_path.name})"
            except Exception:
                pass

        path.write_text(content, encoding=encoding)
        line_count = content.count("\n") + 1
        return {
            "status": "success",
            "output": f"已写入: {path} ({line_count} 行){backup_info}",
            "error_information": ""
        }
    except PermissionError:
        return {"status": "error", "output": "",
                "error_information": f"没有权限写入文件: {path}"}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def edit_file(
    file_path: str,
    old_string: str,
    new_string: str,
) -> Dict[str, Any]:
    """编辑文件: 查找 old_string 并替换为 new_string。

    使用 5 级容错 Replacer 链。当 old_string 为空时创建新文件。
    """
    logger.info(f"[edit_file] 编辑文件: {file_path}")

    try:
        from akg_agents.core_v2.tools.edit_utils import find_and_replace

        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()

        if not old_string:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(new_string, encoding="utf-8")
            line_count = new_string.count("\n") + 1
            return {
                "status": "success",
                "output": f"已创建文件: {path} ({line_count} 行)",
                "error_information": ""
            }

        if not path.exists():
            return {
                "status": "error",
                "output": "",
                "error_information": f"文件不存在: {file_path}"
            }

        content = path.read_text(encoding="utf-8")
        new_content, match_info = find_and_replace(content, old_string, new_string)

        if new_content is None:
            content_lines = content.splitlines()
            old_lines = old_string.splitlines()
            return {
                "status": "error",
                "output": "",
                "error_information": (
                    f"未找到匹配内容。{match_info}\n"
                    f"文件 {path.name} 共 {len(content_lines)} 行, "
                    f"old_string 共 {len(old_lines)} 行。\n"
                    f"提示: 请使用 read_file 确认文件的实际内容后再尝试。"
                )
            }

        backup_path = path.with_suffix(path.suffix + ".bak")
        try:
            path.rename(backup_path)
        except Exception:
            pass

        path.write_text(new_content, encoding="utf-8")

        return {
            "status": "success",
            "output": f"编辑成功: {path} [{match_info}]",
            "error_information": ""
        }

    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"编辑文件失败: {type(e).__name__}: {str(e)}"
        }


def scan_dir(
    path: str = ".",
    max_depth: int = 3,
) -> Dict[str, Any]:
    """浏览目录结构: 列出文件 + Python 文件摘要（函数/类定义）。"""
    dir_path = Path(path).expanduser()
    if not dir_path.is_absolute():
        dir_path = dir_path.resolve()

    if not dir_path.exists():
        return {"status": "error", "output": "", "error_information": f"路径不存在: {dir_path}"}
    if dir_path.is_file():
        return {"status": "error", "output": "",
                "error_information": f"这是文件: {dir_path}\n请使用 read_file 读取。"}

    try:
        results = [f"[DIR] {dir_path}"]
        py_files = []
        other_files = []
        for item in sorted(dir_path.rglob("*")):
            rel = item.relative_to(dir_path)
            if any(part.startswith('.') for part in rel.parts):
                continue
            if "__pycache__" in str(rel):
                continue
            if len(rel.parts) > max_depth:
                continue
            if item.is_file():
                (py_files if item.suffix == ".py" else other_files).append(item)

        if other_files:
            results.append(f"\n Other files ({len(other_files)}):")
            for f in other_files[:30]:
                results.append(f"  {f.relative_to(dir_path)}")

        results.append(f"\n Python files ({len(py_files)}):")
        for f in py_files:
            rel = f.relative_to(dir_path)
            try:
                file_content = f.read_text(encoding="utf-8")
                lines = file_content.count("\n") + 1
                defs = []
                for m in re.finditer(r'^(class |def |async def )(\w+)', file_content, re.MULTILINE):
                    defs.append(f"{m.group(1).strip()} {m.group(2)}")
                defs_str = ", ".join(defs[:15])
                if len(defs) > 15:
                    defs_str += f" ... (+{len(defs)-15})"
                results.append(f"  {rel} ({lines} lines) -> {defs_str}")
            except (UnicodeDecodeError, PermissionError):
                results.append(f"  {rel} (unreadable)")

        return {"status": "success", "output": "\n".join(results), "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def grep_search(
    pattern: str,
    path: str = ".",
    glob: str = "*.py",
    max_results: int = 50,
    context_lines: int = 0,
) -> Dict[str, Any]:
    """在文件/目录中搜索正则表达式。"""
    search_path = Path(path).expanduser()
    if not search_path.is_absolute():
        search_path = search_path.resolve()

    results = []
    try:
        files = [search_path] if search_path.is_file() else sorted(search_path.rglob(glob))
        regex = re.compile(pattern, re.IGNORECASE)
        for f in files:
            try:
                file_lines = f.read_text(encoding="utf-8").splitlines()
                for i, line in enumerate(file_lines):
                    if regex.search(line):
                        full_path = str(f.resolve()).replace('\\', '/')
                        results.append(f"{full_path}:{i+1}: {line.rstrip()}")
                        if context_lines > 0:
                            start = max(0, i - context_lines)
                            end = min(len(file_lines), i + context_lines + 1)
                            for j in range(start, end):
                                if j != i:
                                    results.append(f"  {j+1}| {file_lines[j].rstrip()}")
                        if len(results) >= max_results * (1 + context_lines * 2):
                            break
            except (UnicodeDecodeError, PermissionError):
                continue
            if len(results) >= max_results * (1 + context_lines * 2):
                break
        output = "\n".join(results) if results else "no matches"
        return {"status": "success", "output": output, "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


def list_dir(
    path: str = ".",
) -> Dict[str, Any]:
    """列出目录下所有文件及行数信息。"""
    dir_path = Path(path).expanduser()
    if not dir_path.is_absolute():
        dir_path = dir_path.resolve()

    if not dir_path.exists():
        return {"status": "error", "output": "",
                "error_information": f"目录不存在: {dir_path}"}
    if not dir_path.is_dir():
        return {"status": "error", "output": "",
                "error_information": f"不是目录: {dir_path}"}

    try:
        files = sorted(dir_path.rglob("*"))
        if not files:
            return {"status": "success", "output": "目录为空", "error_information": ""}

        results = [f"[目录] {dir_path}"]
        for f in files:
            if f.is_file():
                rel = f.relative_to(dir_path)
                try:
                    file_lines = f.read_text(encoding="utf-8").count("\n") + 1
                    results.append(f"  {rel} ({file_lines} lines)")
                except Exception:
                    results.append(f"  {rel}")
        return {"status": "success", "output": "\n".join(results), "error_information": ""}
    except Exception as e:
        return {"status": "error", "output": "", "error_information": str(e)}


# ==================== 代码执行工具 ====================


def execute_script(
    script_path: str = "",
    code: str = "",
    args: str = "",
    stdin_input: str = None,
    timeout: int = 60,
    working_dir: str = None,
) -> Dict[str, Any]:
    """执行脚本或内联代码。

    两种模式（二选一）:
    - script_path: 执行文件。自动根据后缀选择 python/bash。
    - code: 将代码写入临时文件后执行（默认 Python）。

    超长输出自动截断。
    """
    if not script_path and not code:
        return {"status": "error", "output": "",
                "error_information": "需要 script_path 或 code 参数（二选一）"}

    logger.info(f"[execute_script] script_path={script_path or '(inline code)'}, args={args}")

    tmp_path = None
    try:
        if working_dir:
            cwd = Path(working_dir).expanduser().resolve()
        else:
            cwd = Path.cwd()

        if script_path:
            path = Path(script_path).expanduser()
            if not path.is_absolute():
                path = path.resolve()

            if not path.exists():
                return {"status": "error", "output": "",
                        "error_information": f"脚本不存在: {script_path}"}
            if not path.is_file():
                return {"status": "error", "output": "",
                        "error_information": f"路径不是文件: {script_path}"}

            suffix = path.suffix.lower()
            if suffix == ".py":
                cmd = [sys.executable, str(path)]
            elif suffix == ".sh":
                cmd = ["bash", str(path)]
            else:
                cmd = [str(path)]
        else:
            tmp = tempfile.NamedTemporaryFile(
                suffix=".py", mode="w", delete=False, encoding="utf-8")
            tmp.write(code)
            tmp.close()
            tmp_path = tmp.name
            cmd = [sys.executable, tmp_path]

        if args:
            import shlex
            cmd.extend(shlex.split(args))

        logger.info(f"[execute_script] 命令: {' '.join(cmd)}, cwd={cwd}")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            input=stdin_input,
        )

        stdout = _truncate_script_output(result.stdout.strip())
        stderr = _truncate_script_output(result.stderr.strip())

        if result.returncode == 0:
            output = "执行成功\n"
            if stdout:
                output += f"\nstdout:\n{stdout}"
            if stderr:
                output += f"\nstderr:\n{stderr}"
            return {"status": "success", "output": output, "error_information": ""}
        else:
            error_info = f"执行失败 (exit code: {result.returncode})\n"
            if stdout:
                error_info += f"\nstdout:\n{stdout}"
            if stderr:
                error_info += f"\nstderr:\n{stderr}"
            return {"status": "error", "output": "", "error_information": error_info}

    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "",
                "error_information": f"执行超时 ({timeout}秒): {script_path or '(inline code)'}"}
    except PermissionError:
        return {"status": "error", "output": "",
                "error_information": f"没有权限执行: {script_path or '(inline code)'}"}
    except Exception as e:
        return {"status": "error", "output": "",
                "error_information": f"执行失败: {type(e).__name__}: {str(e)}"}
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass


# ==================== 交互工具 ====================


def ask_user(message: str) -> Dict[str, Any]:
    logger.info(f"[ask_user] 询问用户: {message}")

    try:
        print(f"\n{'='*60}")
        print(f"Agent 询问:")
        print(f"{message}")
        print(f"{'='*60}")

        user_input = input("您的回复: ").strip()

        logger.info(f"[ask_user] 用户回复: {user_input}")

        return {
            "status": "success",
            "output": f"用户回复: {user_input}",
            "error_information": ""
        }

    except KeyboardInterrupt:
        return {"status": "error", "output": "", "error_information": "用户中断输入"}
    except Exception as e:
        return {"status": "error", "output": "",
                "error_information": f"获取用户输入失败: {str(e)}"}


# ==================== 代码检查工具 ====================


def check_python_code(
    file_path: str,
    auto_format: bool = True,
    fix_in_place: bool = False,
) -> Dict[str, Any]:
    logger.info(f"[check_python_code] 检查文件: {file_path}")

    try:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()

        if not path.exists():
            return {"status": "error", "output": "",
                    "error_information": f"文件不存在: {file_path}"}
        if not path.is_file():
            return {"status": "error", "output": "",
                    "error_information": f"路径不是文件: {file_path}"}

        output_parts = []
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True, text=True, timeout=30)

        if compile_result.returncode == 0:
            output_parts.append("语法检查通过")
        else:
            error_msg = compile_result.stderr.strip()
            output_parts.append(f"语法错误:\n{error_msg}")
            return {"status": "error", "output": "\n".join(output_parts),
                    "error_information": f"Python 语法错误: {error_msg}"}

        if auto_format:
            check_autopep8 = subprocess.run(
                [sys.executable, "-m", "autopep8", "--version"],
                capture_output=True, text=True)

            if check_autopep8.returncode != 0:
                output_parts.append("\nautopep8 未安装，跳过格式化。安装命令: pip install autopep8")
            else:
                autopep8_cmd = [sys.executable, "-m", "autopep8"]
                if fix_in_place:
                    autopep8_cmd.extend(["--in-place", "--aggressive", "--aggressive", str(path)])
                    format_result = subprocess.run(
                        autopep8_cmd, capture_output=True, text=True, timeout=60)
                    if format_result.returncode == 0:
                        output_parts.append("\n代码已格式化并保存到原文件")
                    else:
                        output_parts.append(f"\n格式化失败: {format_result.stderr}")
                else:
                    autopep8_cmd.extend(["--aggressive", "--aggressive", str(path)])
                    format_result = subprocess.run(
                        autopep8_cmd, capture_output=True, text=True, timeout=60)
                    if format_result.returncode == 0:
                        output_parts.append("\n代码格式化成功")
                        output_parts.append(f"\n格式化后的代码:\n{'-'*60}\n{format_result.stdout}\n{'-'*60}")
                    else:
                        output_parts.append(f"\n格式化失败: {format_result.stderr}")

        return {"status": "success", "output": "\n".join(output_parts), "error_information": ""}

    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "", "error_information": f"检查超时: {file_path}"}
    except Exception as e:
        return {"status": "error", "output": "",
                "error_information": f"检查失败: {type(e).__name__}: {str(e)}"}


def check_markdown(
    file_path: str,
    auto_fix: bool = False,
) -> Dict[str, Any]:
    logger.info(f"[check_markdown] 检查文件: {file_path}")

    try:
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = path.resolve()

        if not path.exists():
            return {"status": "error", "output": "",
                    "error_information": f"文件不存在: {file_path}"}
        if not path.is_file():
            return {"status": "error", "output": "",
                    "error_information": f"路径不是文件: {file_path}"}

        output_parts = []
        check_mdlint = subprocess.run(
            ["markdownlint", "--version"],
            capture_output=True, text=True, shell=True)

        if check_mdlint.returncode != 0:
            check_npx = subprocess.run(
                ["npx", "markdownlint-cli", "--version"],
                capture_output=True, text=True, shell=True)
            if check_npx.returncode != 0:
                return {"status": "error", "output": "",
                        "error_information": (
                            "markdownlint-cli 未安装\n"
                            "安装命令: npm install -g markdownlint-cli")}
            mdlint_cmd = ["npx", "markdownlint-cli"]
        else:
            mdlint_cmd = ["markdownlint"]

        if auto_fix:
            mdlint_cmd.extend(["--fix", str(path)])
        else:
            mdlint_cmd.append(str(path))

        result = subprocess.run(
            mdlint_cmd, capture_output=True, text=True, timeout=30, shell=True)

        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0:
            output_parts.append("Markdown 检查通过" + ("，已自动修复" if auto_fix else ""))
            if stdout:
                output_parts.append(f"\n{stdout}")
        else:
            output_parts.append("Markdown 检查发现问题" + ("（部分已修复）" if auto_fix else ""))
            if stdout:
                output_parts.append(f"\n问题列表:\n{stdout}")
            if stderr:
                output_parts.append(f"\n错误信息:\n{stderr}")

        return {
            "status": "success" if result.returncode == 0 else "error",
            "output": "\n".join(output_parts),
            "error_information": "" if result.returncode == 0 else "Markdown 格式检查未通过"
        }

    except subprocess.TimeoutExpired:
        return {"status": "error", "output": "", "error_information": f"检查超时: {file_path}"}
    except FileNotFoundError:
        return {"status": "error", "output": "",
                "error_information": "markdownlint 命令未找到"}
    except Exception as e:
        return {"status": "error", "output": "",
                "error_information": f"检查失败: {type(e).__name__}: {str(e)}"}


# ==================== Skill 加载工具 ====================


def load_skill(name: str, skills_dir: str = None) -> Dict[str, Any]:
    """动态加载 Skill 获取领域指导。

    skills_dir 由调用方注入（如 KernelAgent 注入算子领域的 skills 目录）。
    """
    logger.info(f"[load_skill] 加载 Skill: {name}, skills_dir={skills_dir}")

    try:
        from akg_agents.core_v2.skill.registry import SkillRegistry

        registry = SkillRegistry()

        if skills_dir:
            sd = Path(skills_dir)
            if sd.exists():
                registry.load_from_directory(sd)

        skill = registry.get(name)
        if skill is None:
            available = sorted(registry.get_names())
            return {
                "status": "error",
                "output": "",
                "error_information": (
                    f"Skill '{name}' 不存在。\n"
                    f"可用 Skills: {', '.join(available) if available else '无'}"
                )
            }

        content = skill.content
        if not content:
            return {"status": "error", "output": "",
                    "error_information": f"Skill '{name}' 内容为空"}

        output_parts = [f"[Skill: {name}]"]
        if hasattr(skill, 'description') and skill.description:
            output_parts.append(f"描述: {skill.description}")
        if hasattr(skill, 'recommended_tools') and skill.recommended_tools:
            output_parts.append(f"推荐工具: {', '.join(skill.recommended_tools)}")

        output_parts.append(f"\n{content}")

        if hasattr(skill, 'tool_guidance') and skill.tool_guidance:
            output_parts.append(f"\n---\n工具使用指导:\n{skill.tool_guidance}")

        return {"status": "success", "output": "\n".join(output_parts), "error_information": ""}

    except Exception as e:
        logger.error(f"[load_skill] 加载失败: {e}", exc_info=True)
        return {"status": "error", "output": "",
                "error_information": f"加载 Skill 失败: {str(e)}"}


# ==================== 工具注册 ====================


def _register_all():
    """将 basic tools 注册到统一 ToolRegistry"""
    from akg_agents.core_v2.tools.tool_registry import ToolRegistry

    ToolRegistry.register(
        name="read_file",
        description=(
            "读取文件内容（带行号显示）。\n\n"
            "适用场景:\n"
            "- 查看代码文件、配置文件、日志文件等\n"
            "- 使用 offset/limit 查看大文件的指定区间\n\n"
            "超过 300 行自动截断（可用 offset/limit 翻页）。\n"
            "二进制文件返回文件元信息而不是内容。\n\n"
            "输出: 带行号的文件内容 + 元信息（文件名、总行数、截断提示）。"
        ),
        parameters={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "文件路径（绝对或相对路径）"},
            "offset": {"type": "integer", "description": "起始行号 (1-based)，例如 offset=100 从第100行开始"},
            "limit": {"type": "integer", "description": "读取行数，配合 offset 使用"},
            "encoding": {"type": "string", "description": "文件编码，默认 utf-8", "default": "utf-8"},
        }, "required": ["file_path"]},
        func=read_file,
        category="basic",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="write_file",
        description=(
            "创建或覆盖写入文件。\n\n"
            "自动创建不存在的父目录。覆盖已有文件时会自动创建 .bak 备份。\n\n"
            "输出: 写入确认（文件路径、行数、备份信息）。"
        ),
        parameters={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "目标文件路径"},
            "content": {"type": "string", "description": "要写入的完整文件内容"},
            "overwrite": {"type": "boolean", "description": "是否覆盖已有文件，默认 true", "default": True},
            "encoding": {"type": "string", "description": "文件编码，默认 utf-8", "default": "utf-8"},
        }, "required": ["file_path", "content"]},
        func=write_file,
        category="basic",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="edit_file",
        description=(
            "编辑文件: 查找 old_string 并替换为 new_string。\n\n"
            "适用场景: 修改代码中的特定函数/类/配置、修复 bug、局部修改。\n\n"
            "容错机制: 内置 5 级 Replacer 链（精确匹配 -> 空白容差 -> 锚点相似度 "
            "-> 空白归一化 -> 缩进灵活匹配）。\n"
            "old_string 为空时创建新文件。修改前自动创建 .bak 备份。\n\n"
            "输出: 编辑结果 + 使用的匹配级别。"
        ),
        parameters={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "要编辑的文件路径"},
            "old_string": {"type": "string", "description": "要替换的目标文本（空字符串=创建新文件）"},
            "new_string": {"type": "string", "description": "替换后的文本"},
        }, "required": ["file_path", "old_string", "new_string"]},
        func=edit_file,
        category="basic",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="scan_dir",
        description="浏览目录结构: 列出文件列表 + Python 文件摘要（函数/类定义）。",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "要浏览的目录路径"},
            "max_depth": {"type": "integer", "description": "最大递归深度，默认 3"},
        }, "required": ["path"]},
        func=scan_dir,
        category="basic",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="grep_search",
        description="在文件或目录中使用正则表达式搜索内容。",
        parameters={"type": "object", "properties": {
            "pattern": {"type": "string", "description": "正则表达式模式"},
            "path": {"type": "string", "description": "搜索路径（文件或目录），默认当前目录"},
            "glob": {"type": "string", "description": "文件过滤 glob 模式，默认 '*.py'"},
            "max_results": {"type": "integer", "description": "最大结果数，默认 50"},
            "context_lines": {"type": "integer", "description": "上下文行数，默认 0"},
        }, "required": ["pattern"]},
        func=grep_search,
        category="basic",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="list_dir",
        description="列出指定目录下所有文件及行数信息。",
        parameters={"type": "object", "properties": {
            "path": {"type": "string", "description": "目录路径"},
        }, "required": ["path"]},
        func=list_dir,
        category="basic",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="execute_script",
        description=(
            "执行脚本文件或内联代码。\n\n"
            "两种模式（二选一）:\n"
            "- script_path: 执行脚本文件（.py -> python, .sh -> bash）\n"
            "- code: 直接执行 Python 代码字符串（无需先保存为文件）\n\n"
            "超长输出自动截断。\n\n"
            "输出: stdout 和 stderr 内容。"
        ),
        parameters={"type": "object", "properties": {
            "script_path": {"type": "string", "description": "脚本文件路径（与 code 二选一）"},
            "code": {"type": "string", "description": "Python 代码字符串（与 script_path 二选一）"},
            "args": {"type": "string", "description": "命令行参数", "default": ""},
            "timeout": {"type": "integer", "description": "超时秒数，默认 60", "default": 60},
            "working_dir": {"type": "string", "description": "工作目录"},
        }, "required": []},
        func=execute_script,
        category="execution",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="ask_user",
        description="向用户询问问题并等待回复。用于需要用户提供额外信息或确认的场景。",
        parameters={"type": "object", "properties": {
            "message": {"type": "string", "description": "要询问用户的问题"},
            "task_completed": {
                "type": "boolean",
                "description": (
                    "初始任务是否已完成。任务全部执行完毕后汇报结果时设为 true，"
                    "流程中间需要确认/补充信息时设为 false。"
                ),
                "default": False,
            },
        }, "required": ["message"]},
        func=ask_user,
        category="interaction",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="check_python_code",
        description=(
            "检查并格式化 Python 代码文件。\n"
            "1. 预编译检查语法错误\n"
            "2. 使用 autopep8 自动格式化（符合 PEP 8）"
        ),
        parameters={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "要检查的 Python 文件路径"},
            "auto_format": {"type": "boolean", "description": "是否格式化，默认 true", "default": True},
            "fix_in_place": {"type": "boolean", "description": "是否修改原文件，默认 false", "default": False},
        }, "required": ["file_path"]},
        func=check_python_code,
        category="basic",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="check_markdown",
        description="检查 Markdown 文档格式（使用 markdownlint-cli）。",
        parameters={"type": "object", "properties": {
            "file_path": {"type": "string", "description": "要检查的 Markdown 文件路径"},
            "auto_fix": {"type": "boolean", "description": "是否自动修复，默认 false", "default": False},
        }, "required": ["file_path"]},
        func=check_markdown,
        category="basic",
        scopes=["all"],
    )

    ToolRegistry.register(
        name="load_skill",
        description=(
            "动态加载 Skill 获取领域知识指导。\n\n"
            "Skill 返回 Markdown 格式的领域知识内容，包含编码规范、"
            "示例代码、常见错误和工具使用建议。"
        ),
        parameters={"type": "object", "properties": {
            "name": {"type": "string", "description": "Skill 名称"},
            "skills_dir": {"type": "string", "description": "Skills 目录路径（由调用方注入）"},
        }, "required": ["name"]},
        func=load_skill,
        category="interaction",
        scopes=["all"],
    )


_register_all()
