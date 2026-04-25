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

import subprocess
import logging
import sys
import os
import platform

logger = logging.getLogger(__name__)


def run_command(cmd_list, cmd_msg="untitled_command", env=None, timeout=300, cwd=None):
    """
    运行命令行程序

    Args:
        cmd_list: 命令列表
        cmd_msg: 命令描述信息
        env: 环境变量
        timeout: 超时时间（秒），默认5分钟。设置为None禁用timeout
        cwd: 工作目录。如果指定，命令将在该目录下执行（线程安全，不使用 os.chdir）

    Returns:
        tuple: (是否成功, 错误信息)
    """
    try:
        # 确保环境变量包含Python无缓冲设置
        if env is None:
            env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,  # 改为无缓冲
            env=env,
            cwd=cwd)  # 指定工作目录（线程安全）

        # 根据timeout参数决定是否使用超时
        if timeout is None:
            # 不使用timeout，让进程自然完成
            stdout, stderr = process.communicate()
        else:
            # 使用timeout参数
            try:
                stdout, stderr = process.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                # 超时处理
                process.kill()
                # 等待进程完全结束
                try:
                    stdout, stderr = process.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    # 强制终止
                    process.terminate()
                    stdout, stderr = "", ""

                error_msg = f"Command '{cmd_msg}' timed out after {timeout} seconds"
                logger.error(error_msg)
                return False, error_msg

        # 处理和显示输出
        if stdout:
            for line in stdout.splitlines(True):
                print(line.rstrip())
        if stderr:
            for line in stderr.splitlines(True):
                print(line.rstrip(), file=sys.stderr)

        returncode = process.returncode
        result = (returncode == 0)
        cmd_desc = f"'{cmd_msg}'" if cmd_msg else "Untitled"

        if result:
            logger.info(f"{cmd_desc} executed successfully.")
        else:
            logger.error(f"{cmd_desc} failed with return code {returncode}.")
            if stderr:
                logger.error(f"Error output: {stderr.strip()}")

        return (result, stderr or "")

    except Exception as e:
        logger.error(f"Exception during {cmd_msg} execution: {e}")
        return (False, str(e))
