"""Utilities for running MLIR pass pipelines with verbose logging."""

import io
import os
import re
import threading
from pathlib import Path

from mfusion import ir
from mfusion.passmanager import PassManager
from mfusion.dialects import torch as torch_d

# Thread-safe counter
_pipeline_runner_counter = threading.Lock()
_global_pipeline_id = 0
_global_ir_id = 0

# Subdirectory under the stage IR root for fine-grained (sub-pass) dumps from Python.
VERBOSE_IR_SUBDIR = "verbose_ir"

# When set to "1", internal sub-pass dumps (stage title contains " / ") are emitted only if
# the module IR changed across that pass (stable generic+local-scope text comparison).
_VERBOSE_IR_DUMP_ON_CHANGE_ENV = "MFUSION_VERBOSE_IR_DUMP_ON_CHANGE"


def _verbose_ir_dump_on_change_enabled() -> bool:
    return os.environ.get(_VERBOSE_IR_DUMP_ON_CHANGE_ENV, "").strip() == "1"


def _is_internal_subpass_stage(stage_title: str) -> bool:
    """True for expanded torch-fusion / mfuse-fusion substeps (e.g. 'Torch Fusion / fuse-matmul-cast')."""
    return " / " in stage_title


def _stable_module_ir_text(module: ir.Module) -> str:
    """Text form comparable across a pass run (aligns with generic + local scope IR dump)."""
    op = module.operation
    try:
        buf = io.StringIO()
        op.print(file=buf, print_generic_op_form=True, use_local_scope=True)
        return buf.getvalue()
    except TypeError:
        try:
            buf = io.StringIO()
            op.print(buf, print_generic_op_form=True, use_local_scope=True)
            return buf.getvalue()
        except (TypeError, AttributeError):
            return str(module)
    except AttributeError:
        return str(module)


def _ir_env_level(env_name: str) -> int:
    """Parse MFUSION_* verbosity: unset/other -> 0, '1' -> stage-only, '2' -> stage + internal verbose."""
    v = os.environ.get(env_name, "").strip()
    if v == "2":
        return 2
    if v == "1":
        return 1
    return 0


def _parse_mlir_module_from_text(text: str) -> ir.Module:
    """Parse MLIR module from text IR."""
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    return ir.Module.parse(text, ctx)


def _get_safe_filename(pipeline_id: int, step: int, stage: str) -> str:
    """Convert stage description into a safe filename.

    Example:
        pipeline_id=0, step=0, "Original MLIR Module"
        -> "00_pm00_original_mlir_module_0000.mlir"
    """
    global _global_ir_id
    with _pipeline_runner_counter:
        current_id = _global_ir_id
        _global_ir_id += 1

    name = stage.lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    filename = f"{pipeline_id:02d}_pm{step:02d}_{name}_{current_id:04d}.mlir"
    return filename


def _get_save_directory() -> Path:
    """Get the directory where IR files will be stored.

    If MFUSION_SAVE_IR_PATH is not set, a 'graphs' directory
    will be created under the current working directory.
    """
    configured = os.environ.get("MFUSION_SAVE_IR_PATH")
    if configured:
        path = Path(configured)
        try:
            path.mkdir(parents=True, exist_ok=True)
            return path
        except (OSError, PermissionError) as e:
            print(f"Warning: Failed to create configured save directory {path}: {e}")
            # Fallback to default

    default_path = Path(os.getcwd()) / "graphs"
    try:
        default_path.mkdir(parents=True, exist_ok=True)
        return default_path
    except (OSError, PermissionError) as e:
        print(f"Warning: Failed to create default save directory {default_path}: {e}")
        # Fallback to current directory
        return Path.cwd()


def get_verbose_ir_directory() -> Path:
    """Directory for optional fine-grained IR files under ``<save_root>/verbose_ir/``."""
    d = _get_save_directory() / VERBOSE_IR_SUBDIR
    try:
        d.mkdir(parents=True, exist_ok=True)
        return d
    except (OSError, PermissionError) as e:
        print(f"Warning: Failed to create verbose IR directory {d}: {e}")
        return _get_save_directory()


class PipelineRunner:
    """Run MLIR pass pipelines and optionally dump IR between stages.

    MFUSION_PRINT_IR / MFUSION_SAVE_IR:
      - ``1``: print/save IR only at each pipeline *stage* (coarse).
      - ``2``: same as ``1``, plus internal sub-pass print/save for composite stages
        (``torch-fusion``, ``mfuse-fusion``) implemented in Python by running each
        sub-pass as its own ``run()`` so IR is emitted after every internal pass.
        Decompose / Recompose are pattern-only; they do not emit nested dumps at level ``2``.

    MFUSION_VERBOSE_IR_DUMP_ON_CHANGE:
      - When set to ``1``, internal sub-pass stages (titles containing ``" / "``) only
        print/save if the module IR changed for that pass (stable generic+local-scope text).

    The runner keeps an internal step counter starting from 0. When verbose printing is
    enabled (level >= 1), each stage title is prefixed with the current step as two digits.
    """

    def __init__(self, module: ir.Module):
        global _global_pipeline_id
        with _pipeline_runner_counter:
            self._pipeline_id = _global_pipeline_id
            _global_pipeline_id += 1

        self._module = module
        self._step = 0
        self._print_level = _ir_env_level("MFUSION_PRINT_IR")
        self._save_level = _ir_env_level("MFUSION_SAVE_IR")
        self.enabled_print_ir = self._print_level >= 1
        self.enabled_save_ir = self._save_level >= 1
        # Fine-grained (sub-pass) dumps when MFUSION_PRINT_IR=2 or MFUSION_SAVE_IR=2.
        self.enabled_verbose_internal_ir = self._print_level >= 2 or self._save_level >= 2
        # Use a fixed title for the initial module print/save.
        self._print_and_maybe_save("Original MLIR Module")

    @staticmethod
    def from_torch_dialect_str(torch_dialect_str: str) -> "PipelineRunner":
        """Create a runner from Torch dialect MLIR text.

        This parses the text into an MLIR module, prints the original module
        when verbose printing is enabled, and returns a runner instance with
        the module stored internally.
        """
        module = _parse_mlir_module_from_text(torch_dialect_str)
        runner = PipelineRunner(module)
        return runner

    def _print_and_maybe_save(self, stage_title: str) -> bool:
        """Print or save the current IR if enabled. Returns True on success."""
        success = True

        # Print IR to stdout when enabled.
        if self.enabled_print_ir:
            try:
                print()
                print("=" * 80)
                print(f"{self._step:02d}", stage_title)
                print("=" * 80)
                print(self._module)
                print()
            except (OSError, UnicodeError) as e:
                print(f"Warning: Failed to print IR: {e}")
                success = False

        # Save IR to file when enabled.
        if self.enabled_save_ir:
            try:
                if self._save_level >= 2 and _is_internal_subpass_stage(stage_title):
                    save_dir = get_verbose_ir_directory()
                else:
                    save_dir = _get_save_directory()
                filename = _get_safe_filename(self._pipeline_id, self._step, stage_title)
                path = save_dir / filename
                with path.open("w", encoding="utf-8") as f:
                    f.write(str(self._module))
            except (OSError, IOError) as e:
                print(f"Warning: Failed to save IR to file: {e}")
                success = False

        return success

    @property
    def module(self) -> ir.Module:
        """Return the underlying MLIR module."""
        return self._module

    def run(self, pipeline: str, stage: str) -> ir.Module:
        """Run a pass pipeline on the internal module and print the result.

        Returns:
            The modified MLIR module.

        Raises:
            RuntimeError: If pipeline execution fails.
        """
        self._step += 1
        skip_dump_if_unchanged = (
            _verbose_ir_dump_on_change_enabled()
            and _is_internal_subpass_stage(stage)
            and (self.enabled_print_ir or self.enabled_save_ir)
        )
        before_ir = (
            _stable_module_ir_text(self._module) if skip_dump_if_unchanged else None
        )

        with self._module.context:
            pm = PassManager.parse(pipeline)
            try:
                pm.run(self._module.operation)
            except Exception as e:
                raise RuntimeError(f"Pipeline execution failed: {e}") from e

        if skip_dump_if_unchanged:
            after_ir = _stable_module_ir_text(self._module)
            if after_ir == before_ir:
                return self._module

        # Always try to print/save, even if pipeline had warnings
        self._print_and_maybe_save(stage)
        return self._module

    def run_passes(self, passes: list, stage: str) -> ir.Module:
        """Run a list of passes and print the result.

        Args:
            passes: List of pass names or pass strings.
            stage: Stage name for output.

        Returns:
            The modified MLIR module.
        """
        pipeline_str = "builtin.module(" + ",".join(passes) + ")"
        return self.run(pipeline_str, stage)

    def get_current_step(self) -> int:
        """Get the current step number."""
        return self._step

    def reset_step_counter(self) -> None:
        """Reset the step counter to 0."""
        self._step = 0
