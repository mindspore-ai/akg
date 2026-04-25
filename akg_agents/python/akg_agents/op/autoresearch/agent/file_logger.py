"""
FileLogger — stdout tee to agent.log.

Extracted from SessionStore so the stdout-hijacking lifecycle
is an explicit, self-contained resource.
"""

import os
import sys
import time


class _TeeWriter:
    """Write to two streams simultaneously (stdout + log file)."""

    def __init__(self, stdout, log_file):
        self._stdout = stdout
        self._log_file = log_file
        self._at_line_start = True

    def write(self, data):
        self._stdout.write(data)
        if not data:
            return
        lines = data.split("\n")
        for i, line in enumerate(lines):
            if i > 0:
                self._log_file.write("\n")
                self._at_line_start = True
            if line:
                if self._at_line_start:
                    ts = time.strftime("%H:%M:%S")
                    self._log_file.write(f"[{ts}] {line}")
                else:
                    self._log_file.write(line)
                self._at_line_start = False
        if data.endswith("\n"):
            self._at_line_start = True
        self._log_file.flush()

    def flush(self):
        self._stdout.flush()
        self._log_file.flush()

    def __getattr__(self, name):
        return getattr(self._stdout, name)


class FileLogger:
    """Manages stdout tee to agent.log.  Use as a context manager or call open/close."""

    def __init__(self, task_dir: str):
        self._task_dir = task_dir
        self._log_file = None
        self._original_stdout = None

    def open(self):
        log_path = os.path.join(self._task_dir, "agent.log")
        try:
            self._log_file = open(log_path, "a", encoding="utf-8")
        except OSError:
            try:
                with open(log_path, "w", encoding="utf-8") as f:
                    pass
                self._log_file = open(log_path, "a", encoding="utf-8")
            except OSError as e:
                print(f"[AgentLoop] WARNING: cannot create agent.log ({e}), logging to stdout only")
                self._log_file = None
                return
        self._original_stdout = sys.stdout
        sys.stdout = _TeeWriter(sys.stdout, self._log_file)

    def close(self):
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            self._original_stdout = None
        if self._log_file and not self._log_file.closed:
            self._log_file.close()
            self._log_file = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
