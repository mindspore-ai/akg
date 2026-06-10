# Copyright 2025-2026 Huawei Technologies Co., Ltd
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

"""One-shot SSH probe — gathers structured facts about the remote.

Returns raw facts only. Classification (severity / suggestion) belongs to
``diagnostics.py``; this layer doesn't decide what's fatal vs warning so
the same probe can be reused under different DSL contexts (e.g. triton
missing is fatal for triton_ascend, warn for ascendc_catlass).

Single SSH round-trip; ``stdout=PIPE, stderr=DEVNULL`` avoids the Windows
subprocess+capture_output deadlock that bit OpenSSH / PowerShell."""

from __future__ import annotations

import shlex
import subprocess
from typing import Optional

from .remote_env import source_env_var_bash


# Probe checks are split per concern so each can be classified
# independently. The bash here only emits ``KEY:value`` markers + a
# ``LOG_TAIL_BEGIN`` sentinel; no shell-side severity logic.
#
# arch detection matches the requested probe device when provided and
# falls back to the first visible device otherwise. Name spelling is
# normalized by ``arch_normalize`` rather than a hard-coded SKU table.
# import 检查走 exit code 而不是 tail 输出 —— CANN 初始化可能往 stderr
# 写 LOG_WARNING（如日志目录权限），即使 import 成功也会污染最后一行，
# 用 tail -1 会被误判 fatal。捕获 stdout+stderr 到 var，先看 $? 再决定
# 报 ok 还是 stderr 尾行。
_PROBE_BASH = r"""
env_script={env_script}
repo_path={repo_path}
probe_device={probe_device}
port={port}
log_file={log_file}
echo "ENV_PATH:$env_script"
echo "ENV_OK:$([ -n "$env_script" ] && [ -f "$env_script" ] && echo yes || echo no)"
{env_setup}
if [ -n "$repo_path" ] && [ -d "$repo_path/akg_agents/python" ]; then
  export PYTHONPATH="$repo_path/akg_agents/python:${{PYTHONPATH:-}}"
fi
TORCH_NPU_OUT=$(python -c 'import torch_npu' 2>&1); TORCH_NPU_RC=$?
if [ $TORCH_NPU_RC -eq 0 ]; then
  echo "TORCH_NPU:ok"
else
  echo "TORCH_NPU:$(echo "$TORCH_NPU_OUT" | tail -1)"
fi
TRITON_OUT=$(python -c 'import triton' 2>&1); TRITON_RC=$?
if [ $TRITON_RC -eq 0 ]; then
  echo "TRITON:ok"
else
  echo "TRITON:$(echo "$TRITON_OUT" | tail -1)"
fi
echo "NPU_SMI:$(command -v npu-smi >/dev/null 2>&1 && echo ok || echo missing)"
echo "PROBE_DEVICE:$probe_device"
ASCEND_CHIP="$(npu-smi info 2>/dev/null | awk -v did="$probe_device" '/^\| +[0-9]+ +[0-9A-Z]/{{if (did == "" || $2 == did) {{print $3; exit}}}}')"
echo "ARCH:$(ARCH_NAME="$ASCEND_CHIP" python -c 'import os; from akg_agents.op.utils.arch_normalize import normalize_ascend_arch_name; print(normalize_ascend_arch_name(os.environ.get("ARCH_NAME","")) or "")' 2>/dev/null)"
echo "DEVICES:$(npu-smi info 2>/dev/null | grep -cE '^\| +[0-9]+ +[0-9A-Z]')"
echo "NVIDIA_SMI:$(command -v nvidia-smi >/dev/null 2>&1 && echo ok || echo missing)"
if [ -n "$probe_device" ]; then
  CUDA_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "$probe_device" 2>/dev/null | head -1)"
else
  CUDA_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 2>/dev/null | head -1)"
fi
echo "CUDA_NAME:$CUDA_NAME"
echo "CUDA_ARCH:$(CUDA_NAME="$CUDA_NAME" python -c 'import os; from akg_agents.op.utils.arch_normalize import normalize_cuda_arch_name; print(normalize_cuda_arch_name(os.environ.get("CUDA_NAME","")) or "")' 2>/dev/null)"
echo "CUDA_DEVICES:$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
echo "CPU_ARCH:$(python -c 'from akg_agents.op.utils.arch_normalize import normalize_cpu_arch_name; print(normalize_cpu_arch_name() or "")' 2>/dev/null)"
echo "PORT_PID:$(lsof -ti :$port -sTCP:LISTEN 2>/dev/null | head -1)"
# 远端磁盘剩余空间（MB）—— /tmp + / 中较小那个。daemon 起来后会写日志、
# Python logging flush 失败 → "Logging error" cascade 把 stderr 灌爆，
# 上游应该 fatal 拦截。BusyBox / Alpine 的 df 不支持 --output；走 awk
# 解析第 4 列（Available KB），多目标取最小。
echo "DISK_FREE_MB:$(df -kP /tmp / 2>/dev/null | awk 'NR>1 {{print int($4/1024)}}' | sort -n | head -1)"
echo "LOG_TAIL_BEGIN"
[ -f "$log_file" ] && tail -20 "$log_file" || echo "(no log: $log_file)"
"""


def _first_device_id(device_ids) -> Optional[int]:
    if device_ids is None:
        return None
    if isinstance(device_ids, str):
        parts = [p.strip() for p in device_ids.split(",") if p.strip()]
        return int(parts[0]) if parts else None
    if isinstance(device_ids, (list, tuple, set)):
        return int(next(iter(device_ids))) if device_ids else None
    return int(device_ids)


def probe_remote(ssh_alias: str, env_script: Optional[str], port: int,
                 log_file: Optional[str] = None,
                 repo_path: Optional[str] = None,
                 device_ids=None) -> dict:
    """One SSH round-trip → dict of raw facts.

    Returned keys:
      - ``_SSH_ERROR``: present iff SSH transport itself failed
      - ``ENV_PATH``: configured env_script path (empty if None passed in)
      - ``ENV_OK``: "yes" if path configured AND file exists, "no" otherwise
      - ``TORCH_NPU``: "ok" or Python error string (last traceback line)
      - ``TRITON``: "ok" or Python error string
      - ``NPU_SMI``: "ok" or "missing"
      - ``PROBE_DEVICE``: selected device used for arch inference, or ""
      - ``ARCH``: normalized Ascend arch token or "" if not detected
      - ``DEVICES``: chip count as string (e.g. "8")
      - ``NVIDIA_SMI``: "ok" or "missing"
      - ``CUDA_ARCH``: normalized CUDA arch token (e.g. "a100")
      - ``CUDA_DEVICES``: CUDA device count as string
      - ``CPU_ARCH``: normalized CPU arch token (e.g. "x86_64")
      - ``PORT_PID``: pid string of remote :port LISTEN owner, or ""
      - ``DISK_FREE_MB``: min(/tmp, /) free MB as string; 0 if df failed
      - ``LOG_TAIL``: tail of ``log_file`` or default
        ``/tmp/akg_worker_<port>.log``"""
    log = log_file or f"/tmp/akg_worker_{port}.log"
    probe_device = _first_device_id(device_ids)
    probe = _PROBE_BASH.format(
        env_script=shlex.quote(env_script or ""),
        repo_path=shlex.quote(repo_path or ""),
        probe_device=shlex.quote("" if probe_device is None else str(probe_device)),
        port=port,
        log_file=shlex.quote(log),
        env_setup=source_env_var_bash("env_script"),
    )
    # stderr 走 PIPE 而不是 DEVNULL —— SSH 透传失败时（VPN 没开 / 网络
    # 不通 / 免密配置错 / host alias 不存在）错误只在 stderr 出现，丢
    # 了诊断就只剩"tunnel 起失败"一行很难用。这条 ssh 是 run-and-exit
    # （非 -f 后台），不会触发 Windows -f 那种 pipe-deadlock。
    # ConnectTimeout 限定 10s 让 SSH 早早放弃（默认要等 60s+），加
    # BatchMode=yes 禁掉密码 prompt（CI / 非交互场景下别卡 stdin）。
    # 不加 LogLevel=ERROR：那会连"Connection closed"/"timed out" 这类
    # INFO 级诊断也一起压掉，probe 这边正需要它们。
    try:
        out = subprocess.run(
            ["ssh",
             "-o", "ConnectTimeout=10",
             "-o", "BatchMode=yes",
             ssh_alias, f"bash -lc {shlex.quote(probe)}"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"_SSH_ERROR": "ssh probe 30s 超时（env.sh source / torch_npu "
                              "import 可能无响应；手动 ssh 进去跑一遍）"}
    except Exception as e:
        return {"_SSH_ERROR": str(e)[:200]}

    if out.returncode != 0:
        # SSH 透传失败：connect / auth / host alias 没找到。stderr 通常
        # 写一两行人话（"Connection timed out" / "Permission denied
        # (publickey)" / "Could not resolve hostname xxx"），直接回给
        # classify，让诊断表里第一行就把根因报清楚。
        err = (out.stderr or "").strip() or f"ssh exit rc={out.returncode}"
        return {"_SSH_ERROR": err[:200]}

    facts: dict = {}
    log_lines: list = []
    in_log = False
    for line in out.stdout.splitlines():
        if in_log:
            log_lines.append(line)
        elif line == "LOG_TAIL_BEGIN":
            in_log = True
        elif ":" in line:
            k, v = line.split(":", 1)
            facts[k] = v.strip()
    facts["LOG_TAIL"] = "\n".join(log_lines)
    return facts
