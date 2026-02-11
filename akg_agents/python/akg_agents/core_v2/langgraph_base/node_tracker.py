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

"""通用 LangGraph 节点追踪装饰器

提供节点执行追踪功能，支持：
- 执行时间记录
- 可选的流式消息发送
- 灵活的配置选项
"""

import functools
import logging
import time
from typing import Any, Callable, Dict

logger = logging.getLogger(__name__)


def track_node(node_name: str, require_session: bool = False, require_task_label: bool = False):
    """节点追踪装饰器（通用版本）
    
    为 LangGraph 节点函数添加追踪功能：
    - 记录执行时间
    - 可选的 session_id 校验
    - 可选的流式消息发送
    
    Args:
        node_name: 节点名称，用于日志和消息
        require_session: 是否要求 session_id（默认 False）
        require_task_label: 是否要求 task_label（默认 False）
    
    Example:
        @track_node("my_node")
        async def my_node(state: MyState) -> dict:
            # 节点逻辑
            return {"result": "done"}
    """
    def decorator(node_fn: Callable):
        @functools.wraps(node_fn)
        async def wrapped(state: Dict[str, Any]):
            session_id = str(state.get("session_id") or "").strip()
            task_label = str(state.get("task_label") or "").strip()
            
            if require_task_label and not task_label:
                raise ValueError(f"[{node_name}] state 中必须包含 task_label")
            
            start = time.time()
            
            # 发送开始消息（如果有 session）
            if session_id:
                _safe_send_start(session_id, node_name)
            
            try:
                result = await node_fn(state)
                # 发送节点完成摘要（如果有 session）
                if session_id and isinstance(result, dict):
                    _safe_send_result(session_id, node_name, result)
                return result
            finally:
                elapsed = time.time() - start
                logger.debug(f"[{node_name}] completed in {elapsed:.2f}s")
        
        return wrapped
    return decorator


def _safe_send_start(session_id: str, node_name: str):
    """安全发送开始消息
    
    发送失败不影响主流程，仅记录警告日志。
    """
    if not session_id:
        return
    try:
        from akg_agents.cli.runtime.message_sender import send_message
        from akg_agents.cli.messages import DisplayMessage
        send_message(session_id, DisplayMessage(text=f"▶ {node_name}"))
    except Exception as e:
        logger.warning(f"[{node_name}] send_message failed: {e}")


def _safe_send_result(session_id: str, node_name: str, result: Dict[str, Any]):
    """安全发送节点执行结果摘要
    
    根据节点类型提取关键结果信息并发送到 CLI。
    发送失败不影响主流程，仅记录警告日志。
    """
    if not session_id:
        return
    
    summary = _build_result_summary(node_name, result)
    if not summary:
        return
    
    try:
        from akg_agents.cli.runtime.message_sender import send_message
        from akg_agents.cli.messages import DisplayMessage
        send_message(session_id, DisplayMessage(text=summary))
    except Exception as e:
        logger.warning(f"[{node_name}] send result message failed: {e}")


def _extract_compile_errors(error_log: str) -> str:
    """从编译/验证日志中提取真正有用的错误信息
    
    ninja 构建失败时，原始日志通常包含大量 traceback 和 ninja 调用信息，
    但真正的根因（如编译错误、语法错误、未定义符号等）往往被淹没。
    本函数尝试提取最有诊断价值的部分。
    
    Returns:
        提取后的错误摘要字符串
    """
    import re
    
    lines = error_log.splitlines()
    
    # 策略1: 提取 C/C++ 编译错误（gcc/g++/clang 格式: file:line:col: error: ...）
    compile_errors = []
    for line in lines:
        if re.search(r':\d+:\d+:\s*(error|fatal error):', line):
            compile_errors.append(line.strip())
    
    # 策略2: 提取 Python 错误（最后一个 Exception/Error 行）
    python_errors = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        # 匹配 Python 异常行，如 "SyntaxError: ...", "NameError: ...", "TypeError: ..."
        if re.match(r'^(\w+Error|\w+Exception|RuntimeError|CalledProcessError):', stripped):
            python_errors.append(stripped)
        # 匹配 "raise XxxError" 后面跟的实际错误消息
        elif re.match(r'^(subprocess\.CalledProcessError|torch\.utils\.cpp_extension\.)', stripped):
            python_errors.append(stripped)
    
    # 策略3: 提取 "error:" 或 "Error:" 关键行（通用匹配）
    generic_errors = []
    for line in lines:
        stripped = line.strip()
        if stripped and re.search(r'\berror\b', stripped, re.IGNORECASE):
            # 排除 traceback 中的 "File ..." 行和纯路径行
            if not stripped.startswith('File ') and not stripped.startswith('Traceback'):
                generic_errors.append(stripped)
    
    # 策略4: 提取验证结果不匹配信息（精度问题）
    precision_errors = []
    for line in lines:
        stripped = line.strip()
        if any(kw in stripped.lower() for kw in ['mismatch', 'tolerance', 'allclose', 'assert', 'not equal', 'max diff']):
            precision_errors.append(stripped)
    
    # 组装结果：优先展示编译错误 > 精度错误 > Python 错误
    parts = []
    
    if compile_errors:
        parts.append("  [编译错误]")
        # 最多展示 5 条编译错误
        for err in compile_errors[:5]:
            parts.append(f"    {err}")
        if len(compile_errors) > 5:
            parts.append(f"    ... 共 {len(compile_errors)} 条编译错误")
    
    if precision_errors:
        parts.append("  [精度错误]")
        for err in precision_errors[:3]:
            parts.append(f"    {err}")
    
    if python_errors and not compile_errors:
        # 如果有编译错误，Python 异常通常只是 CalledProcessError 的包装，不需要重复展示
        parts.append("  [Python 异常]")
        for err in python_errors[:3]:
            parts.append(f"    {err}")
    
    if parts:
        return "\n".join(parts)
    
    # 兜底：如果上述策略都没提取到有用信息，取最后 N 行非空内容
    # （错误信息通常在日志末尾）
    non_empty_lines = [l.strip() for l in lines if l.strip()]
    if non_empty_lines:
        tail = non_empty_lines[-8:]  # 取最后 8 行
        return "  [日志末尾]\n" + "\n".join(f"    {l}" for l in tail)
    
    return ""


def _build_result_summary(node_name: str, result: Dict[str, Any]) -> str:
    """根据节点类型构建结果摘要"""
    
    if node_name == "verifier":
        passed = result.get("verifier_result", False)
        if passed:
            return "  ✅ 验证通过"
        else:
            error = result.get("verifier_error", "")
            if error:
                # 智能提取错误关键信息，而非粗暴截取前 N 字符
                extracted = _extract_compile_errors(error)
                if extracted:
                    summary = f"  ❌ 验证失败:\n{extracted}"
                else:
                    # 提取失败时回退到截取末尾
                    tail_lines = [l for l in error.splitlines() if l.strip()][-8:]
                    error_tail = "\n".join(f"    {l.strip()}" for l in tail_lines)
                    summary = f"  ❌ 验证失败:\n  [日志末尾]\n{error_tail}"
                
                # 提示用户查看完整日志的正确位置
                cur_path = result.get("cur_path", "")
                if cur_path:
                    summary += f"\n\n  💡 完整验证日志见工作目录: {cur_path}"
                else:
                    summary += "\n\n  💡 完整日志可在 agent 工作目录中查看"
                return summary
            return "  ❌ 验证失败（无详细错误信息）"
    
    if node_name == "conductor":
        decision = result.get("conductor_decision", "")
        suggestion = result.get("conductor_suggestion", "")
        if decision or suggestion:
            parts = []
            if decision:
                parts.append(f"  决策: {decision}")
            if suggestion:
                # 截取建议的前 300 字符
                preview = suggestion[:300]
                if len(suggestion) > 300:
                    preview += "..."
                parts.append(f"  建议: {preview}")
            return "\n".join(parts)
    
    # 其他节点不发送摘要
    return ""

