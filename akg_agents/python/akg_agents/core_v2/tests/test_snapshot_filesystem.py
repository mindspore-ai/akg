#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
Snapshot Filesystem Unit Tests

Cross-platform tests for the Snapshot Filesystem implementation.
Can be run directly with: python test_snapshot_filesystem.py
Or with pytest: pytest test_snapshot_filesystem.py -v

Tests cover:
1. Save/Load code files
2. Node state copy (with hardlink optimization)
3. Workspace switch (snapshot restore)
4. File listing and export
"""

import atexit
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# ============================================================================
# Cross-platform setup: Add project root to path and mock unavailable modules
# ============================================================================
_project_root = Path(__file__).parent.parent.parent.parent.absolute()
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Mock langchain_core to avoid import errors if not installed
if "langchain_core" not in sys.modules:
    sys.modules["langchain_core"] = MagicMock()
    sys.modules["langchain_core.embeddings"] = MagicMock()
    sys.modules["langchain_core.pydantic_v1"] = MagicMock()

# Now import the modules under test
from akg_agents.core_v2.filesystem.state import FileSystemState
from akg_agents.core_v2.filesystem.models import (
    NodeState,
    ActionRecord,
    ActionHistoryFact,
    ThinkingState,
    PendingToolsState,
    PendingTool
)

# ============================================================================
# Global cleanup registry for test directories
# ============================================================================
_cleanup_dirs = []


def _register_cleanup(path):
    """Register a directory for cleanup at exit."""
    _cleanup_dirs.append(path)


def _cleanup_all():
    """Cleanup all registered test directories."""
    for path in _cleanup_dirs:
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception:
                pass  # Best effort cleanup


atexit.register(_cleanup_all)


# ============================================================================
# Test Base Class
# ============================================================================
class SnapshotTestCase(unittest.TestCase):
    """Base class for Snapshot Filesystem tests with automatic cleanup."""

    def setUp(self):
        """Create a unique temp directory for each test."""
        self.task_id = f"test_{int(time.time())}_{id(self)}"
        self.base_dir = Path(tempfile.mkdtemp(prefix="akg_snapshot_test_"))
        _register_cleanup(self.base_dir)
        self.fs = FileSystemState(self.task_id, base_dir=str(self.base_dir))
        self.fs.initialize_task()

    def tearDown(self):
        """Clean up test directory."""
        if self.base_dir.exists():
            try:
                shutil.rmtree(self.base_dir)
            except PermissionError:
                # On Windows, retry after a short delay
                time.sleep(0.5)
                try:
                    shutil.rmtree(self.base_dir)
                except Exception:
                    pass  # Will be cleaned up at exit


# ============================================================================
# Core Functionality Tests
# ============================================================================
class TestSnapshotFileSystem(SnapshotTestCase):
    """Core Snapshot Filesystem tests."""

    def test_01_init_creates_directories(self):
        """Test that initialization creates required directories."""
        self.assertTrue(self.fs.task_dir.exists())
        self.assertTrue(self.fs.nodes_dir.exists())
        self.assertTrue(self.fs.workspace_dir.exists())
        self.assertTrue(self.fs.logs_dir.exists())

        # Verify root node
        root_state = self.fs.load_node_state("root")
        self.assertEqual(root_state.node_id, "root")
        self.assertEqual(root_state.status, "init")

        # Verify root code snapshot directory exists
        root_code_dir = self.fs.get_code_snapshot_dir("root")
        self.assertTrue(root_code_dir.exists())

    def test_02_save_and_load_code(self):
        """Test saving and loading code files."""
        code_content = "def hello():\n    print('Hello Snapshot!')"

        # Save code
        self.fs.save_code_file("root", "kernel.py", code_content)

        # Verify code in workspace
        workspace_file = self.fs.workspace_dir / "kernel.py"
        self.assertTrue(workspace_file.exists())

        # Verify code in snapshot directory
        snapshot_file = self.fs.get_code_snapshot_dir("root") / "kernel.py"
        self.assertTrue(snapshot_file.exists())

        # Verify loaded content matches (normalize line endings)
        loaded_content = self.fs.load_code_file("root", "kernel.py")
        self.assertEqual(
            loaded_content.replace('\r\n', '\n'),
            code_content.replace('\r\n', '\n')
        )

    def test_03_node_copy_with_hardlink(self):
        """Test that node copy uses hardlinks when possible."""
        # Create code in root
        self.fs.save_code_file("root", "base.py", "# Base Code")

        # Copy node
        self.fs.copy_node_state("root", "node_a")

        # Verify code snapshot was copied
        node_a_snapshot = self.fs.get_code_snapshot_dir("node_a") / "base.py"
        self.assertTrue(node_a_snapshot.exists())

        # Verify content matches
        loaded = self.fs.load_code_file("node_a", "base.py")
        self.assertEqual(loaded.strip(), "# Base Code")

    def test_04_workspace_switch(self):
        """Test workspace switching between nodes."""
        workspace_file = self.fs.workspace_dir / "kernel.py"

        # Root creates v1
        self.fs.save_code_file("root", "kernel.py", "v1")

        # Fork Node A and modify
        self.fs.copy_node_state("root", "node_a")
        self.fs.set_current_node("node_a")
        self.fs.save_code_file("node_a", "kernel.py", "v2-modified-in-a")

        # Fork Node B from Root (sibling of A)
        self.fs.set_current_node("root")
        self.fs.copy_node_state("root", "node_b")
        self.fs.set_current_node("node_b")
        self.fs.save_code_file("node_b", "kernel.py", "v2-modified-in-b")

        # Verify switching
        self.fs.set_current_node("node_a")
        self.assertEqual(workspace_file.read_text().strip(), "v2-modified-in-a")

        self.fs.set_current_node("node_b")
        self.assertEqual(workspace_file.read_text().strip(), "v2-modified-in-b")

        self.fs.set_current_node("root")
        self.assertEqual(workspace_file.read_text().strip(), "v1")

    def test_05_list_code_files(self):
        """Test listing code files in a node."""
        # Save multiple files
        self.fs.save_code_file("root", "f1.py", "print(1)")
        self.fs.save_code_file("root", "f2.py", "print(2)")
        self.fs.save_code_file("root", "subdir/f3.py", "print(3)")

        # List files
        files = self.fs.list_code_files("root")

        self.assertIn("f1.py", files)
        self.assertIn("f2.py", files)
        # Subdirectory file (path format may vary by OS)
        self.assertTrue(any("f3.py" in f for f in files))

    def test_06_export_node_code(self):
        """Test exporting node code to a directory."""
        # Create node with code
        self.fs.save_code_file("root", "kernel.py", "# Kernel Code")
        self.fs.copy_node_state("root", "export_node")
        self.fs.set_current_node("export_node")
        self.fs.save_code_file("export_node", "test.py", "# Test Code")

        # Export to target directory
        export_dir = self.base_dir / "export_target"
        self.fs.export_node_code("export_node", str(export_dir))

        # Verify exported content
        self.assertTrue((export_dir / "kernel.py").exists())
        self.assertTrue((export_dir / "test.py").exists())
        self.assertEqual((export_dir / "test.py").read_text().strip(), "# Test Code")

    def test_07_metadata_persistence(self):
        """Test node metadata persistence."""
        node_id = "meta_test_node"

        # Create initial state
        state = NodeState(
            node_id=node_id,
            turn=1,
            status="running",
            metrics={"score": 95},
            agent_info={"name": "test_agent"}
        )
        self.fs.save_node_state(node_id, state)

        # Verify persistence
        loaded_state = self.fs.load_node_state(node_id)
        self.assertEqual(loaded_state.node_id, node_id)
        self.assertEqual(loaded_state.metrics["score"], 95)

        # Update state
        self.fs.update_node_state(node_id, status="completed", metrics={"score": 100})
        updated_state = self.fs.load_node_state(node_id)
        self.assertEqual(updated_state.status, "completed")
        self.assertEqual(updated_state.metrics["score"], 100)
        self.assertEqual(updated_state.agent_info["name"], "test_agent")

    def test_08_action_history(self):
        """Test action history management."""
        node_id = "action_test_node"
        self.fs.ensure_dir(self.fs.get_actions_dir(node_id))

        # Create and save history
        history = ActionHistoryFact(node_id=node_id, parent_node_id="root", turn=1)
        action1 = ActionRecord(action_id="a1", tool_name="tool1", result={"out": 1})
        history.add_action(action1)

        self.fs.save_action_history_fact(node_id, history)

        # Load history
        loaded_history = self.fs.load_action_history_fact(node_id)
        self.assertEqual(len(loaded_history.actions), 1)
        self.assertEqual(loaded_history.actions[0].tool_name, "tool1")

        # Append action
        action2 = ActionRecord(action_id="a2", tool_name="tool2", result={"out": 2})
        self.fs.append_action(node_id, action2, parent_node_id="root", turn=1)

        reloaded_history = self.fs.load_action_history_fact(node_id)
        self.assertEqual(len(reloaded_history.actions), 2)
        self.assertEqual(reloaded_history.actions[1].action_id, "a2")

    def test_09_thinking_state(self):
        """Test Thinking state management."""
        node_id = "thinking_test_node"
        self.fs.ensure_dir(self.fs.get_node_dir(node_id))

        thinking = ThinkingState(node_id=node_id, turn=1)
        thinking.add_decision("I should sleep", "sleep()")

        self.fs.save_thinking(node_id, thinking)

        loaded = self.fs.load_thinking(node_id)
        self.assertEqual(loaded.latest_thinking, "I should sleep")
        self.assertEqual(len(loaded.decision_history), 1)

    def test_10_pending_tools(self):
        """Test pending tools management."""
        node_id = "pending_test_node"
        self.fs.ensure_dir(self.fs.get_actions_dir(node_id))

        tool = PendingTool(tool_call_id="call_1", tool_name="search", arguments={"q": "snapshot"})

        self.fs.add_pending_tool(node_id, tool, turn=1)

        loaded = self.fs.load_pending_tools(node_id)
        self.assertEqual(len(loaded.pending_tools), 1)
        self.assertEqual(loaded.pending_tools[0].tool_name, "search")
        self.assertEqual(loaded.pending_tools[0].status, "pending")

        # Mark completed
        self.fs.mark_tool_completed(node_id, "call_1")
        loaded_v2 = self.fs.load_pending_tools(node_id)
        self.assertEqual(loaded_v2.pending_tools[0].status, "completed")

    def test_11_system_prompts(self):
        """Test system prompts management."""
        node_id = "prompt_test_node"

        self.fs.save_system_prompt(node_id, turn=0, prompt="You are a helper.")
        self.fs.save_system_prompt(node_id, turn=1, prompt="You are a coder.")

        loaded_0 = self.fs.load_system_prompt(node_id, 0)
        self.assertEqual(loaded_0, "You are a helper.")

        latest = self.fs.get_latest_system_prompt(node_id)
        self.assertEqual(latest, "You are a coder.")

    def test_12_force_reinitialize(self):
        """Test force reinitialization."""
        # Add some data
        state = NodeState(node_id="node_001", turn=1, status="running")
        self.fs.save_node_state("node_001", state)
        self.fs.save_code_file("node_001", "test.py", "# test")

        # Verify data exists
        self.assertTrue(self.fs.node_exists("node_001"))

        # Force reinitialize
        self.fs.initialize_task(force=True)

        # node_001 should not exist
        self.assertFalse(self.fs.node_exists("node_001"))
        # root should exist
        self.assertTrue(self.fs.node_exists("root"))


# ============================================================================
# Edge Case Tests
# ============================================================================
class TestSnapshotFileSystemEdgeCases(SnapshotTestCase):
    """Edge case tests for Snapshot Filesystem."""

    def test_01_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with self.assertRaises(FileNotFoundError):
            self.fs.load_code_file("root", "nonexistent.py")

    def test_02_nested_directories(self):
        """Test nested directory structure."""
        # Save file in nested directory
        self.fs.save_code_file("root", "a/b/c/deep.py", "# Deep file")

        # Verify load
        loaded = self.fs.load_code_file("root", "a/b/c/deep.py")
        self.assertEqual(loaded.strip(), "# Deep file")

        # Verify listing
        files = self.fs.list_code_files("root")
        self.assertTrue(any("deep.py" in f for f in files))

    def test_03_empty_node_snapshot(self):
        """Test node with no code files."""
        # Create a node without any code
        state = NodeState(node_id="empty_node", turn=1, status="running")
        self.fs.save_node_state("empty_node", state)

        # List should return empty
        files = self.fs.list_code_files("empty_node")
        self.assertEqual(files, [])

    def test_04_overwrite_file(self):
        """Test overwriting existing file."""
        self.fs.save_code_file("root", "kernel.py", "v1")
        self.fs.save_code_file("root", "kernel.py", "v2")

        loaded = self.fs.load_code_file("root", "kernel.py")
        self.assertEqual(loaded.strip(), "v2")


# ============================================================================
# API Compatibility Tests
# ============================================================================
class TestAPICompatibility(SnapshotTestCase):
    """Tests to ensure API compatibility with the original interface."""

    def test_file_state_key_format(self):
        """Test that file_state uses code/ prefix."""
        state = NodeState(node_id="node_001", turn=1, status="running")
        self.fs.save_node_state("node_001", state)

        code = "__global__ void kernel() { }"
        self.fs.save_code_file("node_001", "kernel.cu", code)

        loaded_state = self.fs.load_node_state("node_001")
        self.assertIn("code/kernel.cu", loaded_state.file_state)

    def test_file_state_has_size_field(self):
        """Test that file_state includes size field."""
        state = NodeState(node_id="node_001", turn=1, status="running")
        self.fs.save_node_state("node_001", state)

        code = "__global__ void kernel() { }"
        self.fs.save_code_file("node_001", "kernel.cu", code)

        loaded_state = self.fs.load_node_state("node_001")
        self.assertEqual(loaded_state.file_state["code/kernel.cu"]["size"], len(code))

    def test_copy_node_state_copies_code(self):
        """Test that copy_node_state also copies code files."""
        source_state = NodeState(node_id="node_003", turn=1, status="running")
        self.fs.save_node_state("node_003", source_state)
        self.fs.save_code_file("node_003", "kernel.cu", "source code")

        # Copy node
        self.fs.copy_node_state("node_003", "node_004")

        # Verify code was copied
        copied_code = self.fs.load_code_file("node_004", "kernel.cu")
        self.assertEqual(copied_code, "source code")


# ============================================================================
# Main Entry Point
# ============================================================================
if __name__ == "__main__":
    # Run tests with verbose output
    print("=" * 60)
    print("Snapshot Filesystem Unit Tests")
    print("=" * 60)
    print()

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    suite.addTests(loader.loadTestsFromTestCase(TestSnapshotFileSystem))
    suite.addTests(loader.loadTestsFromTestCase(TestSnapshotFileSystemEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestAPICompatibility))

    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print()
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 60)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
