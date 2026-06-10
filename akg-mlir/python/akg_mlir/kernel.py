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
""" Module for akg support ascend_npu_ir test """
import os
import sys
import logging
import json

from .backends.ascend import ascend_compile

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | os.RTLD_GLOBAL)
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
from .ascend_launch import get_host_functions, get_device_function, torch_launch
sys.setdlopenflags(flags)

logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s] %(asctime)s [%(filename)s:%(lineno)d] %(message)s')


class Kernel:
    """ Kernel for support ascend_npu_ir """
    def __init__(self, kernel_name, torch_path=None, kernel_meta=None):
        self.kernel_name = kernel_name
        self.dynamic = kernel_meta.get('dynamic')
        self.arch = kernel_meta.get('device_name')
        self.work_dir = os.getenv("KERNEL_META_DIR", default="akg_kernel_meta")
        self.torch_path = str(torch_path)
        self.launcher = None

    @staticmethod
    def _write_mlir(input_mlir, mlir_path):
        with open(mlir_path, "w", encoding="utf-8") as f:
            f.write(input_mlir)
        return mlir_path

    @staticmethod
    def get_block_dim(meta_json_path):
        """Get block dimension from the metadata JSON file."""
        if not meta_json_path:
            raise ValueError("meta_json_path is required")
        with open(meta_json_path, "r", encoding="utf-8") as f:
            return int(json.load(f)["blockDim"])

    def get_launcher(self, block_dim, lib_path):
        """Bind launch to ``lib_path`` (absolute path to ``.o`` / ``lib*.so``). Caller must supply it
        when the kernel was loaded from cache and never ran :meth:`compile` in this process."""
        if not lib_path:
            raise ValueError("lib_path is required (path to compiled Ascend binary)")
        if self.dynamic:
            kernel_func, tiling_func, _, tiling_size = get_host_functions(self.kernel_name, lib_path)
        else:
            kernel_func = get_device_function(self.kernel_name, lib_path)
            tiling_func = 0
            tiling_size = 0
        def kernel_call(*args, stream=None):
            torch_launch(self.kernel_name, self.torch_path, kernel_func, tiling_func, tiling_size,
                         block_dim, stream, self.dynamic, *args)

        return kernel_call


    def compile(
        self,
        input_mlir: str,
        work_dir: str = None,
        need_compile: bool = True,
        debug: bool = False,
    ):
        """Compile MLIR under ``work_dir`` or the default output directory.

        All intermediates and bishengir outputs live in that directory:

        - ``{work_dir}/{kernel_name}.mlir`` — input written for ``akg-opt``
        - ``{work_dir}/{kernel_name}_out.mlir`` — ``akg-opt`` output
        - ``{work_dir}/lib{kernel_name}.so`` (dynamic) or ``{work_dir}/{kernel_name}.o``
          — ``bishengir-compile`` ``-o`` target; ``{work_dir}/{kernel_name}.json`` meta
        """
        if not work_dir:
            work_dir = self.work_dir
        work_dir = os.path.abspath(work_dir)
        self.work_dir = work_dir
        os.makedirs(work_dir, exist_ok=True)

        kernel_bin_name = f"lib{self.kernel_name}.so" if self.dynamic else f"{self.kernel_name}.o"
        input_file = os.path.join(work_dir, f"{self.kernel_name}.mlir")
        binary_file = os.path.join(work_dir, kernel_bin_name)

        if need_compile:
            Kernel._write_mlir(input_mlir, input_file)

            dump_ir_path = os.path.join(work_dir, f"{self.kernel_name}.log") if debug else None
            try:
                ascend_compile(
                    input_file=input_file,
                    output_file=binary_file,
                    dyn_shape=self.dynamic,
                    arch=self.arch,
                    dump_ir=debug,
                    mlir_timing=True,
                    dump_ir_path=dump_ir_path
                )
                logging.debug("compile finish, binary save to %s", os.path.abspath(binary_file))
            except Exception as compile_err:
                raise RuntimeError(
                    f"compile MLIR failed, error message: {str(compile_err)}"
                ) from compile_err

        meta_json_path = os.path.join(work_dir, f"{self.kernel_name}.json")
        block_dim = Kernel.get_block_dim(meta_json_path)

        self.launcher = self.get_launcher(block_dim, binary_file)

    def run(self, *args, **kwargs):
        """ launch .so file by akg_ascend_backend. """
        try:
            self.launcher(*args, **kwargs)
            logging.debug("success launch kernel: %s", {self.kernel_name})
        except Exception as running_err:
            raise RuntimeError(
                f"exec {self.kernel_name}.so error, error msg: {str(running_err)}"
            ) from running_err
