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
from pathlib import Path

from langchain.tools import tool
from langgraph.types import interrupt

from akg_agents.core.tools.tool_schemas import (
    AskUserInput,
    ExecuteScriptInput,
    FinishInput,
    ReadFileInput,
    WriteFileInput,
)

logger = logging.getLogger(__name__)

@tool("ask_user", args_schema=AskUserInput)
def ask_user(message: str) -> str:
    """向用户询问问题并等待回复。
    
    使用场景：
    - 需要用户提供额外信息补充
    - 需要用户确认某个操作（如确认 task_desc）
    - 需要向用户展示当前进度并等待反馈
    """
    logger.info(f"ask_user: {message}")

    # 关键：不要使用 print()/input() 抢占 stdin（会导致 prompt_toolkit TUI 卡死/崩溃）。
    # 使用 LangGraph 的 interrupt 机制实现“可恢复暂停”：
    # - 首次调用会中断图执行，并把 message 返回给客户端
    # - 客户端下一轮通过 Command(resume=...) 提供回复后，interrupt 会返回该回复
    try:
        user_input = interrupt(message)
    except KeyError as e:
        # 保护：如果脱离 LangGraph runtime 直接调用 tool（无 pregel scratchpad），给出更可读的错误
        raise RuntimeError(
            "ask_user 必须在 LangGraph 运行时（带 checkpointer）中执行，才能使用 interrupt/resume 机制。"
        ) from e

    try:
        user_input_str = str(user_input).strip()
    except Exception:
        user_input_str = ""

    logger.info(f"ask_user 用户回复: {user_input_str}")
    return f"用户回复: {user_input_str}"


@tool("finish", args_schema=FinishInput)
def finish(final_answer: str, success: bool = True) -> str:
    """标记任务完成并返回最终结果。
    
    使用场景：
    - 已经完成了用户的所有要求
    - 准备向用户展示最终结果
    """
    logger.info(f"finish: success={success}")
    logger.info(f"Final answer: {final_answer[:100]}...")
    return final_answer


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


@tool("read_file", args_schema=ReadFileInput)
def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """读取指定文件的内容。
    
    使用场景：
    - 读取 SKILL.md 获取 Skill 的完整指导（如 `resources/skills/kernel-workflow/SKILL.md`）
    - 读取配置文件、代码文件
    - 读取日志或输出文件
    
    路径解析：
    - `resources/...` 开头的路径会基于项目根目录解析
    - 绝对路径直接使用
    - 其他相对路径相对于当前工作目录
    
    返回：成功时返回文件内容，失败时返回 [ERROR] 开头的错误信息
    """
    logger.info(f"read_file: {file_path}")
    
    try:
        path = _resolve_resource_path(file_path)
        
        if not path.exists():
            return f"[ERROR] 文件不存在: {file_path} (解析为: {path})"
        
        if not path.is_file():
            return f"[ERROR] 路径不是文件: {file_path}"
        
        content = path.read_text(encoding=encoding)
        logger.info(f"read_file 成功: {path}, 大小: {len(content)} 字符")
        return content
        
    except PermissionError:
        return f"[ERROR] 没有权限读取文件: {file_path}"
    except UnicodeDecodeError as e:
        return f"[ERROR] 文件编码错误 ({encoding}): {str(e)}"
    except Exception as e:
        return f"[ERROR] 读取文件失败: {str(e)}"


DEFAULT_OUTPUT_DIR = "./akg_agents_outputs"

# 文件类型到默认文件名的映射
FILE_TYPE_NAMES = {
    "task_desc": "task_desc.py",
    "kernel": "kernel.py",
    "triton": "kernel.py",
    "torch": "task_desc.py",
}


def _resolve_file_path(
    file_path: str = None,
    op_name: str = None,
    file_type: str = "kernel"
) -> Path:
    """
    解析文件路径，支持默认路径生成
    
    Args:
        file_path: 用户指定的路径（如果提供则直接使用）
        op_name: 算子名称（用于生成默认目录）
        file_type: 文件类型（task_desc 或 kernel）
        
    Returns:
        解析后的 Path 对象
    """
    if file_path:
        return Path(file_path).expanduser().resolve()
    
    # 默认目录: ./akg_agents_outputs/{op_name}/
    base_dir = Path(DEFAULT_OUTPUT_DIR)
    
    if op_name:
        base_dir = base_dir / op_name
    else:
        base_dir = base_dir / "unnamed"
    
    filename = FILE_TYPE_NAMES.get(file_type, "output.py")
    
    return (base_dir / filename).resolve()


@tool("write_file", args_schema=WriteFileInput)
def write_file(
    content: str,
    file_path: str = None,
    op_name: str = None,
    file_type: str = "kernel",
    encoding: str = "utf-8",
    overwrite: bool = False
) -> str:
    """将生成的代码保存到文件，自动递归创建父目录。
    
    使用场景：
    - 用户有保存文件的意愿
    - 保存生成的 task 代码（file_type="task_desc"）
    - 保存生成的 kernel 代码（file_type="kernel"）
    
    默认路径：
    - 目录: ./akg_agents_outputs/{op_name}/
    - task_desc 代码: task_desc.py
    - kernel 代码: kernel.py
    
    特性：
    - 自动递归创建不存在的父目录
    - 默认不覆盖已存在的文件（设置 overwrite=True 可覆盖）
    - 返回保存的完整路径供用户查看
    
    返回：成功时返回保存路径，失败时返回 [ERROR] 开头的错误信息
    """
    path = _resolve_file_path(file_path, op_name, file_type)
    
    logger.info(f"write_file: {path}, op_name={op_name}, file_type={file_type}, overwrite={overwrite}")
    
    try:
        if path.exists() and not overwrite:
            return f"[ERROR] 文件已存在: {path}。如需覆盖请设置 overwrite=True"
        
        parent_dir = path.parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"write_file 自动创建目录: {parent_dir}")
        
        path.write_text(content, encoding=encoding)
        logger.info(f"write_file 成功: {path}, 大小: {len(content)} 字符")
        
        return (
            f"[SUCCESS] 文件已保存!\n"
            f"📁 保存路径: {path}\n"
            f"📄 文件大小: {len(content)} 字符\n"
            f"💡 查看命令: cat {path}"
        )
        
    except PermissionError:
        return f"[ERROR] 没有权限写入文件: {path}"
    except OSError as e:
        return f"[ERROR] 文件系统错误: {str(e)}"
    except Exception as e:
        return f"[ERROR] 写入文件失败: {str(e)}"


@tool("execute_script", args_schema=ExecuteScriptInput)
def execute_script(
    script_path: str,
    args: str = "",
    stdin_input: str = None,
    timeout: int = 60,
    working_dir: str = None
) -> str:
    """执行 Skill 中的脚本文件（Python/Bash）。
    
    使用场景：
    - 执行 Skill 提供的验证脚本（如 check_torch_code.py）
    - 执行代码格式化、转换脚本
    - 执行任何 Skill 目录下的辅助脚本
    
    路径解析：
    - `resources/...` 开头的路径会基于项目根目录解析
    - 绝对路径直接使用
    
    脚本类型：
    - `.py` 文件使用 Python 执行
    - `.sh` 文件使用 Bash 执行
    
    返回：
    - [SUCCESS] 开头：脚本执行成功，包含 stdout
    - [ERROR] 开头：脚本执行失败，包含错误信息
    """
    import subprocess
    import sys
    
    logger.info(f"execute_script: {script_path}, args={args}, timeout={timeout}")
    
    try:
        path = _resolve_resource_path(script_path)
        
        if not path.exists():
            return f"[ERROR] 脚本不存在: {script_path} (解析为: {path})"
        
        if not path.is_file():
            return f"[ERROR] 路径不是文件: {script_path}"
        
        if working_dir:
            cwd = _resolve_resource_path(working_dir)
        else:
            try:
                from akg_agents import get_project_root
                cwd = Path(get_project_root())
            except ImportError:
                cwd = path.parent
        suffix = path.suffix.lower()
        if suffix == ".py":
            cmd = [sys.executable, str(path)]
        elif suffix == ".sh":
            cmd = ["bash", str(path)]
        else:
            cmd = [str(path)]
        
        if args:
            import shlex
            cmd.extend(shlex.split(args))
        
        logger.info(f"execute_script: cmd={cmd}, cwd={cwd}")
        
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
            output = f"[SUCCESS] 脚本执行成功\n"
            output += f"脚本: {script_path}\n"
            if stdout:
                output += f"\n输出:\n{stdout}"
            if stderr:
                output += f"\n警告:\n{stderr}"
            return output
        else:
            output = f"[ERROR] 脚本执行失败 (exit code: {result.returncode})\n"
            output += f"脚本: {script_path}\n"
            if stdout:
                output += f"\n stdout:\n{stdout}"
            if stderr:
                output += f"\n stderr:\n{stderr}"
            return output
            
    except subprocess.TimeoutExpired:
        return f"[ERROR] 脚本执行超时 ({timeout}秒): {script_path}"
    except PermissionError:
        return f"[ERROR] 没有权限执行脚本: {script_path}"
    except Exception as e:
        return f"[ERROR] 脚本执行失败: {type(e).__name__}: {str(e)}"


def create_basic_tools() -> list:
    """
    创建基础 tools
    """
    tools = [
        ask_user,
        finish,
        read_file,
        write_file,
        execute_script,
    ]
    
    logger.info(f"Created {len(tools)} basic tools: {[t.name for t in tools]}")
    return tools
