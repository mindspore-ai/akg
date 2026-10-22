# Copyright 2025-2026 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
"""Backend-neutral AKG compilation and TorchNPU execution interface."""

__all__ = ["Kernel"]

import os
from pathlib import Path
from .backends.ascend import ascend_compile


class Kernel:
    """Keep operator selection and DSL generation in the compiler pipeline."""

    def __init__(self, kernel_name, torch_path=None, kernel_meta=None):
        self.kernel_name = kernel_name
        self.kernel_meta = dict(kernel_meta or {})
        self.dynamic = self.kernel_meta.get("dynamic", False)
        self.arch = self.kernel_meta.get("device_name")
        self.backend = self.kernel_meta.get("akg_backend", "pypto")
        self.allow_fallback = self.backend == "bisheng"
        self.torch_path = str(torch_path) if torch_path is not None else None
        self.work_dir = os.getenv("KERNEL_META_DIR", os.path.expanduser("~/.akg"))
        self._compile_dir = None
        self._artifact = None
        self.launcher = None

    def compile(self, input_mlir, work_dir=None, need_compile=True, debug=False,
                opt_level=2, force_compile=False):
        """Generate the kernel and bind the callable supplied by Ascend."""
        need_compile = need_compile or force_compile
        directory = Path(work_dir or self.work_dir)
        if not work_dir and not need_compile:
            directory = self._compile_dir or directory
        directory.mkdir(parents=True, exist_ok=True)
        input_file = directory / f"{self.kernel_name}.mlir"
        if need_compile:
            input_file.write_text(input_mlir, encoding="utf-8")
        artifact = ascend_compile(
            str(input_file), str(directory), dyn_shape=self.dynamic, arch=self.arch,
            dump_ir=debug, mlir_timing=True, dump_ir_path=str(directory / f"{self.kernel_name}_akg-opt.log"),
            opt_level=opt_level, backend=self.backend, need_compile=need_compile,
            kernel_name=self.kernel_name, torch_path=self.torch_path,
            num_outputs=self.kernel_meta.get("num_outputs"))
        self.launcher = artifact.launcher
        self._compile_dir = directory
        self._artifact = artifact
        return artifact

    def run(self, *args, **kwargs):
        """Invoke the launcher supplied by the selected compiler backend."""
        if self.launcher is None:
            raise RuntimeError("AKG kernel has not been compiled or loaded")
        return self.launcher(*args, **kwargs)
