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

"""Facts → ``list[Finding]`` classifier + rich.Table renderer.

Pure mapping. Takes raw facts produced by ``remote_probe.probe_remote``
and emits structured findings with severity + remediation suggestion.
The split lets the same probe feed different severity policies — e.g.
triton_ascend DSL flags missing triton as fatal, ascendc_catlass flags
the same fact as warn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Finding:
    severity: str  # "ok" / "info" / "warn" / "fatal"
    check: str
    result: str
    suggest: str


def _ssh_suggest(err: str) -> str:
    """根据 ssh stderr 关键字猜常见故障类型，给针对性建议。落空时给
    一条通用的检查清单。"""
    low = err.lower()
    if "could not resolve hostname" in low or "name or service not known" in low:
        return "~/.ssh/config 里 alias 没定义；检查别名拼写或 Host 段"
    if "timed out" in low or "no route to host" in low or "network is unreachable" in low:
        return "网络不通；查 VPN / 路由 / 远端机是否在线"
    if "connection refused" in low:
        return "远端 sshd 没起 或 port 22 被防火墙挡了"
    if "connection closed" in low or "connection reset" in low:
        return "服务器主动断（auth 失败 / MaxSessions 满 / 防火墙拦截）—— 远端 /var/log/auth.log 看具体原因"
    if "permission denied" in low or "publickey" in low:
        return "免密 key 没配；ssh-copy-id 或检查 IdentityFile / authorized_keys"
    if "host key verification failed" in low:
        return "host key 变了；ssh-keygen -R <host> 后再连"
    return "检查 ssh alias / 网络 / 免密配置（手动跑一次 `ssh <alias>` 看错误）"


def classify(facts: dict, port: int, *,
             backend: Optional[str] = None,
             dsl: Optional[str] = None,
             for_start: bool = False) -> List[Finding]:
    """Map probe facts to findings.

    Context flags (all optional; defaults to the conservative "ascend +
    unknown DSL + not-starting" policy):

    - ``backend``: ``"ascend"`` / ``"cuda"`` / ``"cpu"`` / None. When set
      to ``"cuda"`` or ``"cpu"`` we don't flag missing torch_npu / npu-smi
      as fatal — those would just be noise on a non-Ascend worker. None
      treated as ascend (the project's default).
    - ``dsl``: e.g. ``"triton_ascend"``, ``"ascendc_catlass"``, ``"pypto"``.
      Only ``triton_*`` DSLs make missing triton fatal; everyone else
      flags as warn.
    - ``for_start``: True when called from ``dispatch_start`` (i.e. about
      to spawn). Remote :port held becomes fatal then — bind will fail
      anyway. False (default) keeps it warn for diagnostic-only callers."""
    findings: List[Finding] = []

    ssh_err = facts.get("_SSH_ERROR")
    if ssh_err:
        return [Finding("fatal", "ssh", ssh_err[:160], _ssh_suggest(ssh_err))]

    # Resolve policy from context.
    ascendish = backend in (None, "", "ascend")
    cudaish = backend == "cuda"
    cpuish = backend == "cpu"
    needs_triton = (dsl or "").startswith("triton")

    # env_script: distinguish "not configured" (use remote default shell
    # env) from "configured but path missing" (config bug, fatal).
    env_path = facts.get("ENV_PATH") or ""
    env_ok = facts.get("ENV_OK") or "no"
    if not env_path:
        findings.append(Finding(
            "info", "env_script", "未配置",
            "如远端默认 shell 已 source CANN/torch_npu 可不填；否则"
            " config.yaml: remote_worker.hosts.<alias>.env_script",
        ))
    elif env_ok == "yes":
        findings.append(Finding("ok", "env_script", env_path, ""))
    else:
        findings.append(Finding(
            "fatal", "env_script", f"配置为 {env_path} 但文件不存在",
            "改 config.yaml: remote_worker.hosts.<alias>.env_script",
        ))

    # torch_npu: ascend backend hard dependency; non-ascend → warn at most.
    torch_result = facts.get("TORCH_NPU") or ""
    if torch_result == "ok":
        findings.append(Finding("ok", "torch_npu", "importable", ""))
    elif ascendish:
        findings.append(Finding(
            "fatal", "torch_npu", torch_result[:120] or "import 失败",
            "env_script 缺 source CANN set_env.sh，或环境未装 torch_npu",
        ))
    else:
        findings.append(Finding(
            "info", "torch_npu", "not importable", f"backend={backend}, 不需要",
        ))

    # triton: fatal only when target DSL needs it (triton_*).
    triton_result = facts.get("TRITON") or ""
    if triton_result == "ok":
        findings.append(Finding("ok", "triton", "importable", ""))
    else:
        sev = "fatal" if needs_triton else "warn"
        suggest = ("triton_* DSL 必需" if needs_triton
                   else "仅 triton_* DSL 需要；其它 DSL 不装也能跑")
        findings.append(Finding(
            sev, "triton", triton_result[:80] or "import 失败", suggest,
        ))

    # npu-smi: required for ascend backend.
    if facts.get("NPU_SMI") == "ok":
        findings.append(Finding("ok", "npu-smi", "in PATH", ""))
    elif ascendish:
        findings.append(Finding(
            "fatal", "npu-smi", "not in PATH",
            "env_script 缺 source CANN set_env.sh",
        ))
    else:
        findings.append(Finding(
            "info", "npu-smi", "not in PATH", f"backend={backend}, 不需要",
        ))

    # nvidia-smi: required for CUDA backend.
    if facts.get("NVIDIA_SMI") == "ok":
        findings.append(Finding("ok", "nvidia-smi", "in PATH", ""))
    elif cudaish:
        findings.append(Finding(
            "fatal", "nvidia-smi", "not in PATH",
            "CUDA worker 需要 nvidia-smi；检查驱动 / PATH / env_script",
        ))
    else:
        findings.append(Finding(
            "info", "nvidia-smi", "not in PATH", f"backend={backend}, 不需要",
        ))

    # arch: backend-specific canonical token; no hard-coded fallback.
    arch = (facts.get("ARCH") or "").strip()
    if arch:
        findings.append(Finding("ok", "npu arch", arch.lower(), ""))
    elif ascendish:
        findings.append(Finding(
            "warn", "npu arch", "未能从 npu-smi 推断",
            "传 --arch 显式指定（如 ascend910b3、ascend950pr）",
        ))

    cuda_arch = (facts.get("CUDA_ARCH") or "").strip()
    if cuda_arch:
        name = (facts.get("CUDA_NAME") or "").strip()
        result = f"{cuda_arch} ({name})" if name else cuda_arch
        findings.append(Finding("ok", "cuda arch", result, ""))
    elif cudaish:
        findings.append(Finding(
            "warn", "cuda arch", "未能从 nvidia-smi 推断",
            "传 --arch 显式指定（如 a100、h100、rtx4090）",
        ))

    cpu_arch = (facts.get("CPU_ARCH") or "").strip()
    if cpu_arch:
        findings.append(Finding("ok", "cpu arch", cpu_arch, ""))
    elif cpuish:
        findings.append(Finding(
            "warn", "cpu arch", "platform.machine() returned empty",
            "传 --arch 显式指定（如 x86_64、aarch64）",
        ))

    # device count: backend-specific.
    try:
        n = int(facts.get("DEVICES") or "0")
    except ValueError:
        n = 0
    if n > 0:
        findings.append(Finding("ok", "npu devices", f"{n} visible", ""))
    elif ascendish:
        findings.append(Finding(
            "fatal", "npu devices", "0 visible",
            "驱动异常 / `npu-smi info` 跑不通；ssh 进去手动 verify",
        ))

    try:
        cuda_n = int(facts.get("CUDA_DEVICES") or "0")
    except ValueError:
        cuda_n = 0
    if cuda_n > 0:
        findings.append(Finding("ok", "cuda devices", f"{cuda_n} visible", ""))
    elif cudaish:
        findings.append(Finding(
            "fatal", "cuda devices", "0 visible",
            "驱动异常 / `nvidia-smi -L` 跑不通；ssh 进去手动 verify",
        ))

    # disk free (POSIX 远端 /tmp + / 取较小)。daemon 写日志一旦撞 ENOSPC，
    # Python logging flush 失败会触发 "Logging error" cascade 灌爆终端，
    # 用户根本看不到 root cause；提前 fatal 把这条路堵住。阈值 500MB —
    # daemon log + worker_state.json + 各类临时文件加起来够用。
    try:
        free_mb = int(facts.get("DISK_FREE_MB") or "0")
    except ValueError:
        free_mb = 0
    if free_mb >= 500:
        findings.append(Finding("ok", "disk free",
                                f"{free_mb} MB", ""))
    elif free_mb > 0:
        findings.append(Finding(
            "fatal", "disk free", f"only {free_mb} MB",
            "远端磁盘几乎满 —— daemon 写日志会 ENOSPC。清 /tmp、清旧日志后重试",
        ))
    # free_mb == 0 means df 调用失败，不报；不构成 fatal。

    # remote port owner: blocks --start; only informational for status diag.
    port_pid = (facts.get("PORT_PID") or "").strip()
    if not port_pid:
        findings.append(Finding("ok", f"remote :{port}", "free", ""))
    else:
        sev = "fatal" if for_start else "warn"
        suggest = (
            f"daemon 起不来 —— bind 会冲突。`ssh <alias> kill {port_pid}` "
            f"或换 port" if for_start
            else f"别人/残留 daemon — `ssh <alias> kill {port_pid}` 或换 port"
        )
        findings.append(Finding(
            sev, f"remote :{port}", f"held by PID {port_pid}", suggest,
        ))

    return findings


def has_fatal(findings: Iterable[Finding]) -> bool:
    return any(f.severity == "fatal" for f in findings)


def render_findings(findings: Iterable[Finding], log_tail: str = "") -> None:
    """Print findings table + (optional) log tail to stderr via rich."""
    from rich.console import Console
    from rich.table import Table
    color = {"ok": "green", "info": "cyan", "warn": "yellow", "fatal": "red"}
    sym = {"ok": "✓", "info": "ⓘ", "warn": "⚠", "fatal": "✗"}
    console = Console(stderr=True)
    table = Table(title="远端诊断", show_header=True)
    table.add_column("Check", style="cyan")
    table.add_column("", width=2)
    table.add_column("Result")
    table.add_column("Suggestion", style="dim")
    for fnd in findings:
        c = color.get(fnd.severity, "white")
        s = sym.get(fnd.severity, "?")
        table.add_row(fnd.check, f"[{c}]{s}[/{c}]", fnd.result, fnd.suggest)
    console.print(table)
    if log_tail and log_tail.strip() and not log_tail.strip().startswith("(no log"):
        console.print(f"[dim]daemon log tail:[/dim]\n{log_tail}")
