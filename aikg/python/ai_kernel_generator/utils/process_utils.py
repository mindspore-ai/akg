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


def run_command(cmd_list, cmd_msg="untitled_command", env=None):
    try:
        process = subprocess.Popen(
            cmd_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env)
        output_lines = []
        error_lines = []

        if platform.system() != "Windows":
            # 非Windows系统使用select和fcntl进行非阻塞读取
            import select
            import fcntl

            # 设置非阻塞模式
            for pipe in [process.stdout, process.stderr]:
                if pipe:
                    fd = pipe.fileno()
                    fl = fcntl.fcntl(fd, fcntl.F_GETFL)
                    fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

            # 实时读取直到进程结束
            while process.poll() is None:
                ready_pipes, _, _ = select.select([process.stdout, process.stderr], [], [], 0.1)

                for pipe in ready_pipes:
                    if pipe == process.stdout:
                        line = pipe.readline()
                        if line:
                            print(line.rstrip())
                            output_lines.append(line)
                    elif pipe == process.stderr:
                        line = pipe.readline()
                        if line:
                            print(line.rstrip(), file=sys.stderr)
                            error_lines.append(line)
        else:
            # Windows系统直接使用communicate，不实时读取
            stdout, stderr = process.communicate()

            if stdout:
                for line in stdout.splitlines(True):
                    print(line.rstrip())
                    output_lines.append(line)
            if stderr:
                for line in stderr.splitlines(True):
                    print(line.rstrip(), file=sys.stderr)
                    error_lines.append(line)

            # Windows不需要再次调用communicate，直接获取返回码
            returncode = process.returncode
            result = (returncode == 0)
            cmd_desc = f"'{cmd_msg}'" if cmd_msg else "Untitled"
            if result:
                logger.info(f"{cmd_desc} executed successfully.")
            else:
                logger.error(f"{cmd_desc} failed with return code {returncode}.")
            return (result, ''.join(error_lines))

        # 确保捕获剩余输出 (仅用于非Windows系统)
        stdout_remainder, stderr_remainder = process.communicate()
        if stdout_remainder:
            for line in stdout_remainder.splitlines(True):
                print(line.rstrip())
                output_lines.append(line)
        if stderr_remainder:
            for line in stderr_remainder.splitlines(True):
                print(line.rstrip(), file=sys.stderr)
                error_lines.append(line)

        returncode = process.returncode
        result = (returncode == 0)
        cmd_desc = f"'{cmd_msg}'" if cmd_msg else "Untitled"
        if result:
            logger.info(f"{cmd_desc} executed successfully.")
        else:
            logger.error(f"{cmd_desc} failed with return code {returncode}.")
        return (result, ''.join(error_lines))

    except Exception as e:
        logger.error(f"Exception during {cmd_msg} execution: {e}")
        return (False, str(e))
