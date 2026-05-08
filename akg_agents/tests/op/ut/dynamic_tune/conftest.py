"""dynamic_tune unit test fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

_PYTHON_DIR = Path(__file__).resolve().parents[4] / "python"
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))
