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
基础 Tools

"""

import logging
from pathlib import Path

from langchain.tools import tool

from ai_kernel_generator.core.tools.tool_schemas import (
    AskUserInput,
    FinishInput,
    ReadFileInput,
    WriteFileInput,
)

logger = logging.getLogger(__name__)

# 这里是把描述写到了函数体里面，langchain框架的实现会自己获取
@tool("ask_user", args_schema=AskUserInput)
def ask_user(message: str) -> str:
    """向用户询问问题并等待回复。
    
    使用场景：
    - 需要用户提供额外信息补充
    - 需要用户确认某个操作（如确认 task_desc）
    - 需要向用户展示当前进度并等待反馈
    """
    logger.info(f"ask_user: {message}")
    
    print(f"Agent 询问:")
    for line in message.split('\n'):
        print(f"   {line}")
    print(f"{'='*60}")
    
    try:
        user_input = input("请输入您的回复: ").strip()
    except EOFError:
        user_input = "[用户未输入]"
    
    logger.info(f"ask_user 用户回复: {user_input}")
    return f"用户回复: {user_input}"


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


@tool("read_file", args_schema=ReadFileInput)
def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """读取指定文件的内容。
    
    使用场景：
    - 读取 SKILL.md 获取 Skill 的完整指导
    - 读取配置文件、代码文件
    - 读取日志或输出文件
    
    返回：成功时返回文件内容，失败时返回 [ERROR] 开头的错误信息
    """
    logger.info(f"read_file: {file_path}")
    
    try:
        path = Path(file_path)
        
        if not path.exists():
            return f"[ERROR] 文件不存在: {file_path}"
        
        if not path.is_file():
            return f"[ERROR] 路径不是文件: {file_path}"
        
        content = path.read_text(encoding=encoding)
        logger.info(f"read_file 成功: {file_path}, 大小: {len(content)} 字符")
        return content
        
    except PermissionError:
        return f"[ERROR] 没有权限读取文件: {file_path}"
    except UnicodeDecodeError as e:
        return f"[ERROR] 文件编码错误 ({encoding}): {str(e)}"
    except Exception as e:
        return f"[ERROR] 读取文件失败: {str(e)}"


# 默认输出目录
DEFAULT_OUTPUT_DIR = "./aikg_outputs"

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
    
    # 默认目录: ./aikg_outputs/{op_name}/
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
    - 保存生成的 Torch task 代码（file_type="task_desc"）
    - 保存生成的 Triton kernel 代码（file_type="kernel"）
    
    默认路径：
    - 目录: ./aikg_outputs/{op_name}/
    - task_desc 代码: task_desc.py
    - kernel 代码: triton.py
    
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
        
        # 返回完整信息
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


def create_basic_tools() -> list:
    """
    创建基础 tools
    
    注意：call_generate_task_desc（OpTaskBuilder）现在作为子 Agent 注册，
    通过 create_sub_agent_tools() 自动创建 call_op_task_builder tool。
        
    Returns:
        基础 tools 列表
    """
    tools = [
        ask_user,
        finish,
        read_file,
        write_file,
    ]
    
    logger.info(f"Created {len(tools)} basic tools: {[t.name for t in tools]}")
    return tools
