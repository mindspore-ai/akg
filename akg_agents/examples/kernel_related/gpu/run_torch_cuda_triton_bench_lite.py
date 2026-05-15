#!/usr/bin/env python3
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

"""GPU convenience wrapper for the Benchmark Lite runner.

Forces ``--backend gpu`` regardless of caller-supplied arguments.
Supports all modes (correctness, performance, full).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_torch_bench_lite import main as benchmark_main  # noqa: E402


def main(argv: Optional[List[str]] = None) -> None:
    forwarded_argv = []
    if argv is not None:
        forwarded_argv.extend(argv)
    else:
        forwarded_argv.extend(sys.argv[1:])

    # Filter out any existing --backend argument to avoid conflicts
    # Handles both "--backend gpu" (two tokens) and "--backend=gpu" (one token)
    filtered_argv = []
    skip_next = False
    for arg in forwarded_argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--backend":
            skip_next = True
            continue
        if arg.startswith("--backend="):
            continue
        filtered_argv.append(arg)

    filtered_argv.extend(["--backend", "gpu"])
    benchmark_main(filtered_argv)


if __name__ == "__main__":
    main()
