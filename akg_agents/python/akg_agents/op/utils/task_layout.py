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

"""AR task directory layout — single source of truth for the conventions
shared across workflows (v1 op/autoresearch + workspace_autoresearch) and
the verifier.

Lives under op/utils/ (not under any one workflow's adapter package) so
the dependency direction stays workflow → conventions and verifier →
conventions, never workflow → workflow.

================================================================================
1. Per-task layout (under ``ar_tasks/<op>_<ts>_<hex6>/``)
================================================================================

  task_dir/
      reference.py          ← REF_FILE_DEFAULT, PyTorch Model + get_inputs
                              (single file, DSL-agnostic)
      <entry_file>          ← adapter.entry_filename_template, the file
                              the LLM mainly edits (kernel.py for triton /
                               tilelang / pypto / catlass wrapper /
                               ascendc meta-Python; <op>_kernel.cpp for
                               hypothetical pure-C++ DSLs, etc.)
      <project_subtree>/    ← multi-file DSLs only (e.g. catlass_op/);
                              filenames + structure from
                              adapter.kernel_project_files
      task.yaml             ← TaskConfig (ref_file, editable_files, ...);
                              editable_files = [entry_file, *project_files]
                              (a flat YAML list — consumers ask the
                              adapter for the entry file, not index [0])
      program.md / SKILL.md ← agent instructions
      .git/                 ← per-task baseline + per-KEEP commit history
      .ar_state/            ← phase machine, plan, history, report

================================================================================
2. Per-batch layout (under ``<batch_dir>/`` for batch.run.py)
================================================================================

  Single-file DSLs:
      <batch_dir>/
          manifest.yaml | manifest.json
          batch_progress.json
          batch.log
          <ref_dir>/<op_name>_ref.py
          <kernel_dir>/<op_name>_kernel.py

  Multi-file DSLs (e.g. ascendc_catlass — adapter sets
  ``kernel_arg_is_directory = True`` + ``kernel_project_dir_name = "catlass_op"``):
      <batch_dir>/
          manifest.yaml | manifest.json
          <ref_dir>/<op_name>_ref.py
          <kernel_dir>/<op_name>/
              <entry_file>              # adapter.entry_filename_template
              <kernel_project_dir>/     # adapter.kernel_project_dir_name
                  kernel/, include/, src/, CMakeLists.txt, ...

  :func:`resolve_kernel_paths_for_op` is the single owner of the per-DSL
  rule that turns ``<kernel_dir>`` + ``op_name`` into a concrete pair of
  paths; ``batch/discover.py`` and ``batch/manifest.py`` delegate to it.

================================================================================
3. DSL-adapter knobs that drive the layout
================================================================================

The base :class:`DSLAdapter` (``op/verifier/adapters/dsl/base.py``)
exposes the structural metadata; this module documents what each knob
*means* for the layout:

  entry_filename_template (str)
      Format-string filename for the op's entry file — the file the LLM
      mainly works on. ``{op_name}`` slot supported but typically unused
      (single literal "kernel.py" works for every Python-style DSL).
      Pure C++ DSLs override to e.g. ``"{op_name}_kernel.cpp"``.

  kernel_arg_is_directory (bool)
      False (default) → ``--kernel`` is a Python file. The wrapper is the
      kernel; ``editable_files = [entry_file]``.
      True → ``--kernel`` is a directory containing a sibling entry file
      + a per-DSL project subtree. ``editable_files = [entry_file] +
      kernel_project_files``.

  kernel_project_dir_name (Optional[str])
      Subdirectory name (relative to per-op root) holding the project
      subtree when ``kernel_arg_is_directory=True``. catlass uses
      ``"catlass_op"``.

  kernel_project_files (list)
      Path entries (files or directories) that belong to the DSL's
      kernel project besides the entry file — sources, headers, build
      scripts. Single-file DSLs leave empty.

  static_check_via_python_ast (bool)
      True iff the entry file is Python source CodeChecker should
      ``ast.parse``. False for ascendc (meta-Python that exec's into
      string vars but doesn't define ``ModelNew``) and any pure-C++
      adapter. NOT the same as "entry file is .py" — ascendc's entry is
      .py but isn't ast-checkable.

  needs_binary_io (bool)
      True iff the DSL uses file-based tensor I/O (swft).

================================================================================
4. Consumer rules (how to read the entry file / ref file)
================================================================================

Any consumer reading from a task_dir should follow these rules instead
of hardcoding ``"reference.py"`` / ``"kernel.py"`` (or indexing a list):

  Want the reference module path → ``task_dir / config.ref_file``
      ``config.ref_file`` defaults to ``REF_FILE_DEFAULT``.

  Want the LLM-edited entry file → ask the adapter
      ``task_dir / adapter.entry_filename_template.format(op_name=...)``.
      Don't assume Python — check
      ``adapter.static_check_via_python_ast`` for parseability.

  Want the full kernel project for handoff → iterate ``config.editable_files``
      The YAML list captures what scaffold wrote; each entry may be a
      file or a directory; copy / tar / pass as ``--kernel`` accordingly.
"""

from pathlib import Path
from typing import Tuple


# Reference file written by every AR scaffolder. Per-task overridable via
# ``TaskConfig.ref_file`` in task.yaml; this constant is the on-disk
# default for both v1 and WA scaffolders, the verifier's
# ``_materialize_framework_bundle`` target, and any reader that needs to
# find the framework Model before task.yaml is loaded.
REF_FILE_DEFAULT = "reference.py"


def py_stem(name: str) -> str:
    """Strip a trailing ``.py`` extension. Idempotent. Used at the
    eval_kernel / worker / eval_client boundary because eval_kernel's
    CLI takes ``--ref-file <stem>`` (no extension) while
    ``TaskConfig.ref_file`` carries the basename WITH ``.py``."""
    return name[:-3] if name.endswith(".py") else name


def resolve_kernel_paths_for_op(adapter, kernel_dir: Path,
                                op_name: str) -> Tuple[Path, Path]:
    """Per-DSL kernel layout resolver for batch-mode (``<kernel_dir>/<op>``).
    Returns ``(kernel_arg, python_module)``:

    * ``kernel_arg`` is what's passed to ``/autoresearch --kernel`` — a
      file for single-file DSLs, a project directory for multi-file ones.
    * ``python_module`` is always the ``.py`` wrapper batch/verify.py
      needs to compile + import + smoke-run.

    Single-file DSLs (``adapter.kernel_arg_is_directory=False``):
        ``<kernel_dir>/<op>_kernel.py`` — both kernel_arg and python_module.

    Multi-file DSLs (catlass etc.):
        ``<kernel_dir>/<op>/<adapter.kernel_project_dir_name>/`` (kernel_arg)
        + sibling entry file named by ``adapter.entry_filename_template``
        or ``<op>_kernel.py`` (KernelBench legacy)."""
    kernel_dir = Path(kernel_dir)
    if not adapter.kernel_arg_is_directory:
        # Per-DSL entry first (e.g. ``<op>_kernel.cpp`` for a pure-C++
        # DSL whose template formats with {op_name}); fall back to legacy
        # ``<op>_kernel.py`` so existing batch fixtures keep working
        # (default ``kernel.py`` template doesn't include op_name).
        canonical = adapter.entry_filename_template.format(op_name=op_name)
        legacy = f"{op_name}_kernel.py"
        for name in (canonical, legacy):
            py = kernel_dir / name
            if py.is_file():
                return py, py
        raise FileNotFoundError(
            f"neither {canonical} nor {legacy} found under {kernel_dir.name}/")

    subdir = adapter.kernel_project_dir_name
    if not subdir:
        raise ValueError(
            f"DSL adapter {type(adapter).__name__} flags "
            f"kernel_arg_is_directory=True but leaves "
            f"kernel_project_dir_name unset")
    op_root = kernel_dir / op_name
    kernel_arg = op_root / subdir
    if not kernel_arg.is_dir():
        raise FileNotFoundError(
            f"{kernel_arg.relative_to(kernel_dir.parent)} not found")
    canonical = adapter.entry_filename_template.format(op_name=op_name)
    for name in (canonical, f"{op_name}_kernel.py"):
        cand = op_root / name
        if cand.is_file():
            return kernel_arg, cand
    raise FileNotFoundError(
        f"{op_root.name}/ has no sibling {canonical} or {op_name}_kernel.py")
