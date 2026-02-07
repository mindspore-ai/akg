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

import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def _resolve_resource_path(file_path: str) -> Path:
    path = Path(file_path)
    
    if path.is_absolute():
        return path
    
    if file_path.startswith("resources/"):
        try:
            from akg_agents import get_project_root
            project_root = Path(get_project_root())
            resolved = project_root / file_path
            logger.debug(f"Resolved resource path: {file_path} -> {resolved}")
            return resolved
        except ImportError:
            logger.warning("Cannot import get_project_root, using relative path")
    
    return path


def read_file(file_path: str, encoding: str = "utf-8") -> Dict[str, Any]:
    logger.info(f"[read_file] 读取文件: {file_path}")
    
    try:
        path = _resolve_resource_path(file_path)
        
        if not path.exists():
            return {
                "status": "error",
                "output": "",
                "error_information": f"文件不存在: {file_path}"
            }
        
        if not path.is_file():
            return {
                "status": "error",
                "output": "",
                "error_information": f"路径不是文件: {file_path}"
            }
        
        content = path.read_text(encoding=encoding)
        logger.info(f"[read_file] 成功读取: {path}, 大小: {len(content)} 字符")
        
        return {
            "status": "success",
            "output": content,
            "error_information": ""
        }
        
    except PermissionError:
        return {
            "status": "error",
            "output": "",
            "error_information": f"没有权限读取文件: {file_path}"
        }
    except UnicodeDecodeError as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"文件编码错误 ({encoding}): {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"读取文件失败: {str(e)}"
        }


def write_file(
    content: str,
    file_path: str = None,
    op_name: str = None,
    file_type: str = "kernel",
    encoding: str = "utf-8",
    overwrite: bool = False
) -> Dict[str, Any]:
    try:
        if file_path:
            path = Path(file_path).resolve()
        else:
            base_dir = Path("~/.akg_tmp")
            if op_name:
                base_dir = base_dir / op_name
            else:
                base_dir = base_dir / "unnamed"
            
            filename_map = {
                "kernel": "kernel.py",
                "test": "test.py",
                "spec": "spec.md",
            }
            filename = filename_map.get(file_type, "output.txt")
            path = (base_dir / filename).resolve()
        
        logger.info(f"[write_file] 写入文件: {path}, 大小: {len(content)} 字符")
        
        if path.exists() and not overwrite:
            return {
                "status": "error",
                "output": "",
                "error_information": f"文件已存在: {path}。如需覆盖请设置 overwrite=True"
            }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        
        logger.info(f"[write_file] 成功写入: {path}")
        
        return {
            "status": "success",
            "output": f"文件已保存到: {path}",
            "error_information": ""
        }
        
    except PermissionError:
        return {
            "status": "error",
            "output": "",
            "error_information": f"没有权限写入文件: {path}"
        }
    except OSError as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"文件系统错误: {str(e)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"写入文件失败: {str(e)}"
        }


def ask_user(message: str) -> Dict[str, Any]:
    logger.info(f"[ask_user] 询问用户: {message}")
    
    try:
        # 打印问题
        print(f"\n{'='*60}")
        print(f"🤖 Agent 询问:")
        print(f"{message}")
        print(f"{'='*60}")
        
        user_input = input("👤 您的回复: ").strip()
        
        logger.info(f"[ask_user] 用户回复: {user_input}")
        
        return {
            "status": "success",
            "output": f"用户回复: {user_input}",
            "error_information": ""
        }
        
    except KeyboardInterrupt:
        return {
            "status": "error",
            "output": "",
            "error_information": "用户中断输入"
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"获取用户输入失败: {str(e)}"
        }


def check_python_code(
    file_path: str,
    auto_format: bool = True,
    fix_in_place: bool = False
) -> Dict[str, Any]:
    logger.info(f"[check_python_code] 检查文件: {file_path}, auto_format={auto_format}, fix_in_place={fix_in_place}")
    
    try:
        path = _resolve_resource_path(file_path)
        
        if not path.exists():
            return {
                "status": "error",
                "output": "",
                "error_information": f"文件不存在: {file_path}"
            }
        
        if not path.is_file():
            return {
                "status": "error",
                "output": "",
                "error_information": f"路径不是文件: {file_path}"
            }
        
        output_parts = []
        logger.info(f"[check_python_code] 步骤 1: 预编译检查语法")
        compile_result = subprocess.run(
            [sys.executable, "-m", "py_compile", str(path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if compile_result.returncode == 0:
            output_parts.append("✅ 语法检查通过")
        else:
            error_msg = compile_result.stderr.strip()
            output_parts.append(f"❌ 语法错误:\n{error_msg}")
            return {
                "status": "error",
                "output": "\n".join(output_parts),
                "error_information": f"Python 语法错误: {error_msg}"
            }
        if auto_format:
            logger.info(f"[check_python_code] 步骤 2: 使用 autopep8 格式化")
            check_autopep8 = subprocess.run(
                [sys.executable, "-m", "autopep8", "--version"],
                capture_output=True,
                text=True
            )
            
            if check_autopep8.returncode != 0:
                output_parts.append("\n⚠️ autopep8 未安装，跳过格式化")
                output_parts.append("安装命令: pip install autopep8")
            else:
                autopep8_cmd = [sys.executable, "-m", "autopep8"]
                
                if fix_in_place:
                    autopep8_cmd.extend(["--in-place", "--aggressive", "--aggressive", str(path)])
                    format_result = subprocess.run(
                        autopep8_cmd,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if format_result.returncode == 0:
                        output_parts.append("\n✅ 代码已格式化并保存到原文件")
                    else:
                        output_parts.append(f"\n⚠️ 格式化失败: {format_result.stderr}")
                else:
                    autopep8_cmd.extend(["--aggressive", "--aggressive", str(path)])
                    format_result = subprocess.run(
                        autopep8_cmd,
                        capture_output=True,
                        text=True,
                        timeout=60
                    )
                    
                    if format_result.returncode == 0:
                        formatted_code = format_result.stdout
                        output_parts.append("\n✅ 代码格式化成功")
                        output_parts.append("\n📝 格式化后的代码:")
                        output_parts.append("-" * 60)
                        output_parts.append(formatted_code)
                        output_parts.append("-" * 60)
                    else:
                        output_parts.append(f"\n⚠️ 格式化失败: {format_result.stderr}")
        
        return {
            "status": "success",
            "output": "\n".join(output_parts),
            "error_information": ""
        }
        
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "output": "",
            "error_information": f"检查超时: {file_path}"
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"检查失败: {type(e).__name__}: {str(e)}"
        }


def check_markdown(
    file_path: str,
    auto_fix: bool = False
) -> Dict[str, Any]:
    logger.info(f"[check_markdown] 检查文件: {file_path}, auto_fix={auto_fix}")
    
    try:
        path = _resolve_resource_path(file_path)
        
        if not path.exists():
            return {
                "status": "error",
                "output": "",
                "error_information": f"文件不存在: {file_path}"
            }
        
        if not path.is_file():
            return {
                "status": "error",
                "output": "",
                "error_information": f"路径不是文件: {file_path}"
            }
        
        output_parts = []
        check_mdlint = subprocess.run(
            ["markdownlint", "--version"],
            capture_output=True,
            text=True,
            shell=True
        )
        
        if check_mdlint.returncode != 0:
            check_npx = subprocess.run(
                ["npx", "markdownlint-cli", "--version"],
                capture_output=True,
                text=True,
                shell=True
            )
            
            if check_npx.returncode != 0:
                return {
                    "status": "error",
                    "output": "",
                    "error_information": (
                        "markdownlint-cli 未安装\n"
                        "安装命令:\n"
                        "  npm install -g markdownlint-cli\n"
                        "或者:\n"
                        "  npx markdownlint-cli --help"
                    )
                }
            mdlint_cmd = ["npx", "markdownlint-cli"]
        else:
            mdlint_cmd = ["markdownlint"]
        
        # 构建命令
        if auto_fix:
            mdlint_cmd.extend(["--fix", str(path)])
        else:
            mdlint_cmd.append(str(path))
        
        logger.info(f"[check_markdown] 执行命令: {' '.join(mdlint_cmd)}")
        
        # 执行检查
        result = subprocess.run(
            mdlint_cmd,
            capture_output=True,
            text=True,
            timeout=30,
            shell=True
        )
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        if result.returncode == 0:
            if auto_fix:
                output_parts.append("✅ Markdown 检查通过，已自动修复可修复的问题")
            else:
                output_parts.append("✅ Markdown 检查通过，无问题")
            
            if stdout:
                output_parts.append(f"\n{stdout}")
        else:
            if auto_fix:
                output_parts.append("⚠️ Markdown 检查发现问题（部分已修复）")
            else:
                output_parts.append("❌ Markdown 检查发现问题")
            
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
        return {
            "status": "error",
            "output": "",
            "error_information": f"检查超时: {file_path}"
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "output": "",
            "error_information": "markdownlint 命令未找到，请确保已安装 markdownlint-cli"
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"检查失败: {type(e).__name__}: {str(e)}"
        }


def execute_script(
    script_path: str,
    args: str = "",
    stdin_input: str = None,
    timeout: int = 60,
    working_dir: str = None
) -> Dict[str, Any]:
    logger.info(f"[execute_script] 执行脚本: {script_path}, args={args}")
    
    try:
        path = _resolve_resource_path(script_path)
        
        if not path.exists():
            return {
                "status": "error",
                "output": "",
                "error_information": f"脚本不存在: {script_path}"
            }
        
        if not path.is_file():
            return {
                "status": "error",
                "output": "",
                "error_information": f"路径不是文件: {script_path}"
            }
        
        # 确定工作目录
        if working_dir:
            cwd = _resolve_resource_path(working_dir)
        else:
            try:
                from akg_agents import get_project_root
                cwd = Path(get_project_root())
            except ImportError:
                cwd = path.parent
        
        # 确定命令
        suffix = path.suffix.lower()
        if suffix == ".py":
            cmd = [sys.executable, str(path)]
        elif suffix == ".sh":
            cmd = ["bash", str(path)]
        else:
            cmd = [str(path)]
        
        # 添加参数
        if args:
            import shlex
            cmd.extend(shlex.split(args))
        
        logger.info(f"[execute_script] 命令: {' '.join(cmd)}, cwd={cwd}")
        
        # 执行脚本
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            input=stdin_input
        )
        
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        
        if result.returncode == 0:
            output = f"脚本执行成功\n"
            if stdout:
                output += f"\nstdout:\n{stdout}"
            if stderr:
                output += f"\nstderr:\n{stderr}"
            
            return {
                "status": "success",
                "output": output,
                "error_information": ""
            }
        else:
            error_info = f"脚本执行失败 (exit code: {result.returncode})\n"
            if stdout:
                error_info += f"\nstdout:\n{stdout}"
            if stderr:
                error_info += f"\nstderr:\n{stderr}"
            
            return {
                "status": "error",
                "output": "",
                "error_information": error_info
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "output": "",
            "error_information": f"脚本执行超时 ({timeout}秒): {script_path}"
        }
    except PermissionError:
        return {
            "status": "error",
            "output": "",
            "error_information": f"没有权限执行脚本: {script_path}"
        }
    except Exception as e:
        return {
            "status": "error",
            "output": "",
            "error_information": f"脚本执行失败: {type(e).__name__}: {str(e)}"
        }
