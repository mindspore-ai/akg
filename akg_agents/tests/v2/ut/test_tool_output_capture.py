#!/usr/bin/env python3
"""
工具输出捕获测试 - 验证 ToolExecutor 在执行工具时正确捕获 stdout/stderr/logging

运行: cd akg/aikg && source env.sh && python -m pytest tests/v2/ut/test_tool_output_capture.py -v

测试内容:
1. _OutputCapture 基础能力：stdout/stderr/logging 捕获
2. ToolExecutor 执行 basic_tool 时的输出捕获
3. ToolExecutor 执行 domain_tool 时的输出捕获
4. 工具失败/异常时的 error 捕获
5. _save_node_result 保存捕获日志到文件
"""

import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# 确保项目路径可用
project_root = Path(__file__).parent.parent.parent.parent
python_path = project_root / "python"
sys.path.insert(0, str(python_path))

from akg_agents.core_v2.tools.tool_executor import (
    ToolExecutor,
    _OutputCapture,
    _TeeStream,
    _ListHandler,
)


# ==================== _OutputCapture 基础测试 ====================


class TestOutputCapture:
    """测试 _OutputCapture 底层捕获能力"""

    def test_capture_stdout(self):
        """捕获 stdout 输出"""
        with _OutputCapture() as cap:
            print("hello from stdout")
            print("second line")

        assert "hello from stdout" in cap.stdout
        assert "second line" in cap.stdout

    def test_capture_stderr(self):
        """捕获 stderr 输出"""
        with _OutputCapture() as cap:
            sys.stderr.write("error message\n")
            sys.stderr.write("warning: something wrong\n")

        assert "error message" in cap.stderr
        assert "warning: something wrong" in cap.stderr

    def test_capture_logging(self):
        """捕获 logging 输出"""
        test_logger = logging.getLogger("test_capture_logging")
        test_logger.setLevel(logging.DEBUG)

        with _OutputCapture() as cap:
            test_logger.info("info log message")
            test_logger.warning("warning log message")
            test_logger.error("error log message")

        assert "info log message" in cap.logs
        assert "warning log message" in cap.logs
        assert "error log message" in cap.logs

    def test_capture_all_streams(self):
        """同时捕获 stdout + stderr + logging"""
        test_logger = logging.getLogger("test_all_streams")
        test_logger.setLevel(logging.DEBUG)

        with _OutputCapture() as cap:
            print("stdout content")
            sys.stderr.write("stderr content\n")
            test_logger.error("logging content")

        assert "stdout content" in cap.stdout
        assert "stderr content" in cap.stderr
        assert "logging content" in cap.logs

    def test_tee_mode_preserves_original_output(self, capsys):
        """tee 模式：既捕获又不影响原始输出"""
        with _OutputCapture() as cap:
            print("visible output")

        # 捕获了
        assert "visible output" in cap.stdout
        # 原始 stdout 也能看到（capsys 会在 _OutputCapture 外层）
        # 这里主要验证退出后 sys.stdout 恢复正常
        assert sys.stdout is not cap._stdout_buf

    def test_to_dict_empty(self):
        """无输出时 to_dict 返回空字典"""
        with _OutputCapture() as cap:
            pass  # 不产生任何输出

        result = cap.to_dict()
        # 可能有少量 logging 背景噪声，但 stdout/stderr 应为空
        assert "captured_stdout" not in result or result.get("captured_stdout", "") == ""

    def test_to_dict_with_content(self):
        """有输出时 to_dict 返回对应字段"""
        with _OutputCapture() as cap:
            print("stdout here")
            sys.stderr.write("stderr here\n")

        d = cap.to_dict()
        assert "captured_stdout" in d
        assert "stdout here" in d["captured_stdout"]
        assert "captured_stderr" in d
        assert "stderr here" in d["captured_stderr"]

    def test_log_records_list(self):
        """log_records 返回 LogRecord 列表"""
        test_logger = logging.getLogger("test_log_records")
        test_logger.setLevel(logging.DEBUG)

        with _OutputCapture() as cap:
            test_logger.info("record 1")
            test_logger.warning("record 2")

        records = cap.log_records
        assert len(records) >= 2
        messages = [r.getMessage() for r in records]
        assert "record 1" in messages
        assert "record 2" in messages

    def test_restore_streams_on_exception(self):
        """异常时也正确恢复 stdout/stderr"""
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            with _OutputCapture() as cap:
                raise ValueError("test exception")
        except ValueError:
            pass

        assert sys.stdout is original_stdout
        assert sys.stderr is original_stderr


# ==================== ToolExecutor 集成测试 ====================


def _make_executor(tool_types: Dict[str, str] = None) -> ToolExecutor:
    """创建一个测试用的 ToolExecutor（不加载 tools.yaml）"""
    executor = ToolExecutor.__new__(ToolExecutor)
    executor.agent_registry = {}
    executor.workflow_registry = {}
    executor.agent_context = {
        "framework": "torch",
        "backend": "cpu",
        "arch": "x86_64",
        "dsl": "cpp",
        "task_id": "test_capture",
    }
    executor.history = []
    executor.tool_types = tool_types or {}
    return executor


class TestToolExecutorCapture:
    """测试 ToolExecutor.execute() 的输出捕获"""

    @pytest.mark.asyncio
    async def test_basic_tool_stdout_captured(self):
        """basic_tool 产生的 stdout 被捕获到结果中"""
        executor = _make_executor(tool_types={"my_tool": "basic_tool"})

        async def _fake_basic(name, args):
            print(f"[tool_stdout] executing {name}")
            test_logger = logging.getLogger("test.basic")
            test_logger.info(f"[tool_log] {name} started")
            return {"status": "success", "output": "done", "error_information": ""}

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_basic_tool', side_effect=_fake_basic):
                result = await executor.execute("my_tool", {"message": "hello"})

        assert result["status"] == "success"
        assert "captured_stdout" in result
        assert "[tool_stdout] executing my_tool" in result["captured_stdout"]
        assert "captured_logs" in result
        assert "[tool_log] my_tool started" in result["captured_logs"]

    @pytest.mark.asyncio
    async def test_domain_tool_stderr_captured(self):
        """domain_tool 产生的 stderr 被捕获到结果中"""
        executor = _make_executor(tool_types={"my_domain_tool": "domain_tool"})

        async def _fake_domain(name, args):
            sys.stderr.write(f"[WARN] device not found, fallback to cpu\n")
            return {"status": "success", "output": "profiled", "error_information": ""}

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_domain_tool', side_effect=_fake_domain):
                result = await executor.execute("my_domain_tool", {})

        assert result["status"] == "success"
        assert "captured_stderr" in result
        assert "device not found" in result["captured_stderr"]

    @pytest.mark.asyncio
    async def test_tool_error_stderr_fills_error_information(self):
        """工具失败且 error_information 为空时，自动用 captured_stderr 补充"""
        executor = _make_executor(tool_types={"failing_tool": "basic_tool"})

        async def _fake_fail(name, args):
            sys.stderr.write("Segmentation fault (core dumped)\n")
            return {"status": "error", "output": "", "error_information": ""}

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_basic_tool', side_effect=_fake_fail):
                result = await executor.execute("failing_tool", {})

        assert result["status"] == "error"
        # 空的 error_information 应该被 captured_stderr 补充
        assert "Segmentation fault" in result["error_information"]

    @pytest.mark.asyncio
    async def test_tool_existing_error_not_overwritten(self):
        """工具自身已有 error_information 时不被 stderr 覆盖"""
        executor = _make_executor(tool_types={"err_tool": "basic_tool"})

        async def _fake(name, args):
            sys.stderr.write("some background noise\n")
            return {"status": "error", "output": "", "error_information": "specific error"}

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_basic_tool', side_effect=_fake):
                result = await executor.execute("err_tool", {})

        assert result["error_information"] == "specific error"
        # stderr 仍然被捕获
        assert "some background noise" in result.get("captured_stderr", "")

    @pytest.mark.asyncio
    async def test_no_capture_fields_when_silent(self):
        """工具没有额外输出时，不添加 captured_ 字段"""
        executor = _make_executor(tool_types={"quiet_tool": "basic_tool"})

        async def _fake_quiet(name, args):
            # 不产生任何 stdout/stderr 输出
            return {"status": "success", "output": "ok", "error_information": ""}

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_basic_tool', side_effect=_fake_quiet):
                result = await executor.execute("quiet_tool", {})

        assert result["status"] == "success"
        # 可能有 captured_logs（来自 logging 背景），但 stdout/stderr 应该没有
        assert result.get("captured_stdout", "") == ""
        assert result.get("captured_stderr", "") == ""

    @pytest.mark.asyncio
    async def test_exception_in_tool_still_captures(self):
        """工具抛出异常时也能捕获之前的输出"""
        executor = _make_executor(tool_types={"crash_tool": "basic_tool"})

        async def _fake_crash(name, args):
            print("[crash_tool] starting...")
            sys.stderr.write("[crash_tool] about to crash\n")
            raise RuntimeError("unexpected crash")

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_basic_tool', side_effect=_fake_crash):
                # execute 本身会被异常中断，但 _OutputCapture.__exit__ 仍然执行
                # 不过当前 execute 没有 try/except 包裹整个 with 块
                # 异常会传播出来 —— 验证 stream 被正确恢复
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                
                with pytest.raises(RuntimeError, match="unexpected crash"):
                    await executor.execute("crash_tool", {})
                
                # stream 已恢复
                assert sys.stdout is original_stdout
                assert sys.stderr is original_stderr


# ==================== _save_node_result 测试 ====================


class TestSaveNodeResult:
    """测试 _save_node_result 保存捕获日志到文件"""

    def _make_mock_agent(self):
        """创建一个模拟的 ReActAgent（只需要 _save_node_result）"""
        # 直接 import 函数并手动调用
        from akg_agents.core_v2.agents.react_agent import ReActAgent
        
        agent = MagicMock(spec=ReActAgent)
        agent._save_node_result = ReActAgent._save_node_result.__get__(agent)
        return agent

    def test_save_captured_logs_to_files(self, tmp_path):
        """result 中的 captured_* 字段被保存到 logs/ 目录"""
        agent = self._make_mock_agent()

        result = {
            "status": "success",
            "output": "done",
            "error_information": "",
            "captured_stdout": "stdout line 1\nstdout line 2",
            "captured_stderr": "stderr warning here",
            "captured_logs": "INFO test: log message\nERROR test: error happened",
        }

        agent._save_node_result(str(tmp_path), result)

        # 检查文件是否存在
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "logs" / "captured_stdout.txt").exists()
        assert (tmp_path / "logs" / "captured_stderr.txt").exists()
        assert (tmp_path / "logs" / "captured_logs.txt").exists()

        # 检查内容
        assert "stdout line 1" in (tmp_path / "logs" / "captured_stdout.txt").read_text()
        assert "stderr warning here" in (tmp_path / "logs" / "captured_stderr.txt").read_text()
        assert "error happened" in (tmp_path / "logs" / "captured_logs.txt").read_text()

        # result.json 也应包含 captured 字段
        saved_result = json.loads((tmp_path / "result.json").read_text())
        assert saved_result["captured_stdout"] == "stdout line 1\nstdout line 2"

    def test_no_log_dir_when_no_captured(self, tmp_path):
        """没有 captured_* 字段时不创建 logs/ 目录"""
        agent = self._make_mock_agent()

        result = {
            "status": "success",
            "output": "short",
            "error_information": "",
        }

        agent._save_node_result(str(tmp_path), result)

        assert (tmp_path / "result.json").exists()
        # 不应该创建 logs 目录（除非工具自己创建的）
        # 注意：如果 captured_ 字段不存在或为空，不创建
        logs_dir = tmp_path / "logs"
        if logs_dir.exists():
            # 如果存在，不应该有 captured_ 文件
            assert not (logs_dir / "captured_stdout.txt").exists()

    def test_save_with_code_and_logs(self, tmp_path):
        """同时有 generated_code 和 captured 日志的情况"""
        agent = self._make_mock_agent()

        result = {
            "status": "success",
            "output": "Code generated successfully " * 20,  # > 200 chars
            "error_information": "",
            "generated_code": "def add(a, b):\n    return a + b\n",
            "captured_stdout": "Compilation successful",
            "captured_logs": "INFO compiler: compiled in 0.5s",
        }

        agent._save_node_result(str(tmp_path), result)

        # 所有文件都应该存在
        assert (tmp_path / "result.json").exists()
        assert (tmp_path / "code" / "code.py").exists()
        assert (tmp_path / "output.txt").exists()
        assert (tmp_path / "logs" / "captured_stdout.txt").exists()
        assert (tmp_path / "logs" / "captured_logs.txt").exists()

        # 验证代码内容
        assert "def add(a, b):" in (tmp_path / "code" / "code.py").read_text()


# ==================== 模拟真实工具场景 ====================


class TestRealWorldScenarios:
    """模拟真实的 node 识别 / profile 场景"""

    @pytest.mark.asyncio
    async def test_profile_wrong_node_shows_diagnostic_info(self):
        """
        模拟 profile_kernel 读错 node 的场景：
        - node_001 生成了 matmul 的代码
        - node_002 生成了 softmax 的代码
        - profile 应该用 node_002 的代码，但错误地引用了 node_001
        - 验证日志中包含足够的诊断信息
        """
        executor = _make_executor(tool_types={"profile_kernel": "domain_tool"})

        # 模拟 profile_kernel 拿到了错误的代码
        async def _fake_profile(name, args):
            op_name = args.get("op_name", "unknown")
            code = args.get("generated_code", "")

            # 日志记录实际收到的参数（便于诊断 node 引用错误）
            tool_logger = logging.getLogger("akg_agents.tools.profile")
            tool_logger.info(f"[profile_kernel] op_name={op_name}")
            tool_logger.info(f"[profile_kernel] generated_code length={len(code)}")
            tool_logger.info(f"[profile_kernel] code preview: {code[:100]}")

            # 代码不匹配，profiling 失败
            if "softmax" not in code:
                tool_logger.error(
                    f"[profile_kernel] 代码不匹配！期望 softmax 相关代码，"
                    f"但收到的代码包含: {code[:50]}"
                )
                sys.stderr.write(
                    f"ERROR: Code mismatch - expected softmax kernel but got: {code[:50]}...\n"
                )
                return {
                    "status": "fail",
                    "output": "",
                    "error_information": f"Profile 失败: 代码与算子 {op_name} 不匹配",
                    "gen_time_us": None,
                    "base_time_us": None,
                    "speedup": 0.0,
                }

            return {
                "status": "success",
                "output": f"Profile 完成: {op_name}",
                "error_information": "",
                "gen_time_us": 100.0,
                "base_time_us": 120.0,
                "speedup": 1.2,
            }

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_domain_tool', side_effect=_fake_profile):
                # 模拟：错误地传入了 matmul 代码给 softmax 的 profile
                result = await executor.execute("profile_kernel", {
                    "op_name": "softmax",
                    "generated_code": "def matmul_kernel(a, b): return a @ b",
                    "task_code": "import torch; torch.matmul(a, b)",
                })

        # 验证：失败状态 + 捕获的日志/stderr 包含诊断信息
        assert result["status"] == "fail"
        assert "不匹配" in result["error_information"]

        # 关键：captured_logs 包含了实际收到的代码信息
        assert "captured_logs" in result
        assert "code preview:" in result["captured_logs"]
        assert "matmul" in result["captured_logs"]
        assert "代码不匹配" in result["captured_logs"]

        # captured_stderr 也有错误信息
        assert "captured_stderr" in result
        assert "Code mismatch" in result["captured_stderr"]

    @pytest.mark.asyncio
    async def test_verify_kernel_captures_subprocess_output(self):
        """
        模拟 verify_kernel 执行子进程（编译/运行）时的输出捕获
        """
        executor = _make_executor(tool_types={"verify_kernel": "domain_tool"})

        async def _fake_verify(name, args):
            # 模拟子进程产生的输出
            print("[compiler] g++ -O2 -o kernel kernel.cpp")
            print("[compiler] compilation successful")
            print("[runner] executing kernel...")
            print("[runner] result: PASS (tolerance=1e-6)")

            tool_logger = logging.getLogger("akg_agents.tools.verify")
            tool_logger.info(f"[verify_kernel] 验证通过: {args.get('op_name')}")

            return {
                "status": "success",
                "output": "验证通过",
                "error_information": "",
                "verify_log": "[compiler] OK\n[runner] PASS",
            }

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_domain_tool', side_effect=_fake_verify):
                result = await executor.execute("verify_kernel", {
                    "op_name": "add",
                    "generated_code": "void add_kernel(float* a, float* b, float* c, int n) {...}",
                    "task_code": "import torch; c = a + b",
                })

        assert result["status"] == "success"

        # 编译和运行的 stdout 被捕获
        assert "captured_stdout" in result
        assert "compilation successful" in result["captured_stdout"]
        assert "result: PASS" in result["captured_stdout"]

        # logging 也被捕获
        assert "captured_logs" in result
        assert "验证通过" in result["captured_logs"]

    @pytest.mark.asyncio
    async def test_verify_failure_captures_compile_error(self):
        """
        模拟验证失败：编译错误时的完整输出捕获
        """
        executor = _make_executor(tool_types={"verify_kernel": "domain_tool"})

        async def _fake_verify_fail(name, args):
            print("[compiler] g++ -O2 -o kernel kernel.cpp")
            sys.stderr.write(
                "kernel.cpp:42:5: error: use of undeclared identifier 'float4'\n"
                "    float4 vec = load(ptr);\n"
                "    ^\n"
                "1 error generated.\n"
            )

            tool_logger = logging.getLogger("akg_agents.tools.verify")
            tool_logger.error(f"[verify_kernel] 编译失败: {args.get('op_name')}")

            return {
                "status": "fail",
                "output": "",
                "error_information": "编译失败",
                "verify_log": "kernel.cpp:42:5: error: use of undeclared identifier 'float4'",
            }

        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch.object(executor, '_execute_domain_tool', side_effect=_fake_verify_fail):
                result = await executor.execute("verify_kernel", {
                    "op_name": "vectorized_add",
                    "generated_code": "void add(...) { float4 vec = load(ptr); }",
                    "task_code": "import torch; c = a + b",
                })

        assert result["status"] == "fail"

        # stderr 包含完整的编译错误
        assert "captured_stderr" in result
        assert "undeclared identifier 'float4'" in result["captured_stderr"]
        assert "1 error generated" in result["captured_stderr"]

        # logging 也有
        assert "captured_logs" in result
        assert "编译失败" in result["captured_logs"]


# ==================== 端到端整合测试 ====================


class TestEndToEndCapture:
    """端到端测试：execute → result 包含所有 captured 字段"""

    @pytest.mark.asyncio
    async def test_full_pipeline_basic_tool(self):
        """完整 pipeline：basic_tool 执行 + 结果保存"""
        executor = _make_executor()

        # 注册一个会产生各种输出的假 basic_tool
        def noisy_read_file(file_path: str) -> Dict[str, Any]:
            print(f"[read_file] 读取文件: {file_path}")
            logger = logging.getLogger("akg_agents.tools.basic")
            logger.info(f"[read_file] 文件大小: 1234 bytes")
            if not os.path.exists(file_path):
                sys.stderr.write(f"WARNING: file not found: {file_path}\n")
                return {
                    "status": "error",
                    "output": "",
                    "error_information": f"文件不存在: {file_path}",
                }
            return {
                "status": "success",
                "output": "file content here",
                "error_information": "",
            }

        # 直接 patch basic_tools 模块中的函数
        with patch("akg_agents.core_v2.tools.tool_executor.resolve_arguments", side_effect=lambda x: x):
            with patch("akg_agents.core_v2.tools.basic_tools.read_file", noisy_read_file):
                executor.tool_types = {"read_file": "basic_tool"}
                result = await executor.execute("read_file", {
                    "file_path": "/nonexistent/file.txt"
                })

        assert result["status"] == "error"
        assert "文件不存在" in result["error_information"]

        # 验证捕获
        assert "captured_stdout" in result
        assert "读取文件: /nonexistent/file.txt" in result["captured_stdout"]

        assert "captured_stderr" in result
        assert "file not found" in result["captured_stderr"]

        assert "captured_logs" in result
        assert "文件大小: 1234 bytes" in result["captured_logs"]


# ==================== 入口 ====================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
