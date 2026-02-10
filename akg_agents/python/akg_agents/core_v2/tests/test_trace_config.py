import os
import shutil
import pytest
from pathlib import Path
from akg_agents.core_v2.filesystem.state import FileSystemState
from akg_agents.core_v2.filesystem.trace_system import TraceSystem

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path

def test_trace_config_default(temp_dir):
    """Test default behavior (only code/ is linked)"""
    trace = TraceSystem("test_default", base_dir=str(temp_dir))
    trace.initialize()
    
    # Create files in code/ and logs/
    fs = trace.fs
    root_node = "root"
    fs.save_code_file(root_node, "main.py", "print('hello')")
    
    # Manually create a log file (simulate artifact)
    log_dir = fs.get_node_dir(root_node) / "logs"
    log_dir.mkdir()
    (log_dir / "app.log").write_text("log content")
    
    # Create a new node (fork)
    n1 = trace.add_node({"type": "test"}, {"res": "ok"})
    
    # Verify code is copied
    assert fs.load_code_file(n1, "main.py") == "print('hello')"
    
    # Verify logs are NOT copied by default
    n1_dir = fs.get_node_dir(n1)
    assert not (n1_dir / "logs").exists()

def test_trace_config_custom_include(temp_dir):
    """Test including logs/ via .traceconfig"""
    trace = TraceSystem("test_custom", base_dir=str(temp_dir))
    
    # Create .traceconfig
    (temp_dir / ".traceconfig").write_text("code/\nlogs/\n")
    
    trace.initialize() # Should load config
    
    fs = trace.fs
    root_node = "root"
    fs.save_code_file(root_node, "main.py", "print('hello')")
    
    # Manually create a log file
    log_dir = fs.get_node_dir(root_node) / "logs"
    log_dir.mkdir()
    (log_dir / "app.log").write_text("log content")
    
    # Fork
    n1 = trace.add_node({"type": "test"}, {"res": "ok"})
    
    # Verify both code and logs are copied
    assert fs.load_code_file(n1, "main.py") == "print('hello')"
    n1_dir = fs.get_node_dir(n1)
    assert (n1_dir / "logs" / "app.log").exists()
    assert (n1_dir / "logs" / "app.log").read_text() == "log content"

def test_trace_config_exclude(temp_dir):
    """Test excluding .tmp files"""
    trace = TraceSystem("test_exclude", base_dir=str(temp_dir))
    
    # config: include code/, exclude *.tmp
    (temp_dir / ".traceconfig").write_text("code/\n!**/*.tmp\n")
    
    trace.initialize()
    
    fs = trace.fs
    root_node = "root"
    fs.save_code_file(root_node, "main.py", "print('hello')")
    fs.save_code_file(root_node, "temp.tmp", "should be ignored")
    
    # Fork
    n1 = trace.add_node({"type": "test"}, {"res": "ok"})
    
    # Verify
    assert fs.load_code_file(n1, "main.py") == "print('hello')"
    n1_dir = fs.get_node_dir(n1)
    assert not (n1_dir / "code" / "temp.tmp").exists()

def test_trace_config_complex(temp_dir):
    """Test complex patterns (artifacts/*.json)"""
    trace = TraceSystem("test_complex", base_dir=str(temp_dir))
    
    # config: include code/, artifacts/*.json
    (temp_dir / ".traceconfig").write_text("code/\nartifacts/*.json\n")
    
    trace.initialize()
    
    fs = trace.fs
    root_node = "root"
    
    # Create structure:
    # artifacts/result.json
    # artifacts/data.bin (should be ignored)
    rt_dir = fs.get_node_dir(root_node)
    (rt_dir / "artifacts").mkdir()
    (rt_dir / "artifacts" / "result.json").write_text("{}")
    (rt_dir / "artifacts" / "data.bin").write_text("binary")
    
    # Fork
    n1 = trace.add_node({"type": "test"}, {"res": "ok"})
    
    n1_dir = fs.get_node_dir(n1)
    assert (n1_dir / "artifacts" / "result.json").exists()
    assert not (n1_dir / "artifacts" / "data.bin").exists()
