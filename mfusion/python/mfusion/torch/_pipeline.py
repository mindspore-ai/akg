"""Utilities for running MLIR pass pipelines with verbose logging."""

import os
import re
from pathlib import Path

from mfusion import ir
from mfusion.passmanager import PassManager
from mfusion.dialects import torch as torch_d


def _parse_mlir_module_from_text(text: str) -> ir.Module:
    """Parse MLIR module from text IR."""
    ctx = ir.Context()
    torch_d.register_dialect(ctx)
    return ir.Module.parse(text, ctx)


def _get_safe_filename(step: int, stage: str) -> str:
    """Convert stage description into a safe filename.

    Example:
        "Decompose aclnn ops to meta ops"
        -> "02_decompose_aclnn_ops_to_meta_ops.mlir"
    """
    name = stage.lower().replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]", "", name)
    return f"{step:02d}_{name}.mlir"


def _get_save_directory() -> Path:
    """Get the directory where IR files will be stored.

    If MFUSION_SAVE_IR_PATH is not set, a 'graphs' directory
    will be created under the current working directory.
    """
    configured = os.environ.get("MFUSION_SAVE_IR_PATH")
    if configured:
        return Path(configured)
    return Path(os.getcwd()) / "graphs"


class PipelineRunner:
    """Run MLIR pass pipelines and optionally dump IR between stages.

    The runner keeps an internal step counter starting from 0. When verbose
    printing is enabled via the MFUSION_PRINT_IR environment variable, each
    stage title is prefixed with the current step formatted as two digits.
    """

    def __init__(self, module: ir.Module):
        self._module = module
        self._step = 0
        self.enabled_print_ir = os.environ.get("MFUSION_PRINT_IR") == "1"
        self.enabled_save_ir = os.environ.get("MFUSION_SAVE_IR") == "1"
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

    def _print_and_maybe_save(self, stage_title: str):
        """Print or save the current IR if enabled."""
        # Print IR to stdout when enabled.
        if self.enabled_print_ir:
            print()
            print("=" * 80)
            print(f"{self._step:02d}", stage_title)
            print("=" * 80)
            print(self._module)
            print()

        # Save IR to file when enabled.
        if self.enabled_save_ir:
            save_dir = _get_save_directory()
            save_dir.mkdir(parents=True, exist_ok=True)
            # Use the current step for the filename.
            filename = _get_safe_filename(self._step, stage_title)
            path = save_dir / filename
            with path.open("w", encoding="utf-8") as f:
                f.write(str(self._module))

    @property
    def module(self) -> ir.Module:
        """Return the underlying MLIR module."""
        return self._module

    def run(self, pipeline: str, stage: str) -> ir.Module:
        """Run a pass pipeline on the internal module and print the result."""
        self._step += 1
        with self._module.context:
            pm = PassManager.parse(pipeline)
            pm.run(self._module.operation)
        self._print_and_maybe_save(stage)
        return self._module
