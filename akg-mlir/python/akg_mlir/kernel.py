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

from .backends.ascend import ascend_compile, run_akg_opt, dump_ascend_meta_data, get_block_dim_from_mlir

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | os.RTLD_GLOBAL)
# pylint: disable=wrong-import-position
# pylint: disable=wrong-import-order
from akg.akgAscendLaunch import get_host_functions, get_device_function, torch_launch
sys.setdlopenflags(flags)

logging.basicConfig(level=logging.INFO)


class Kernel:
    """ Kernel for support ascend_npu_ir """
    def __init__(self, kernel_name, torch_path=None, kernel_meta=None):
        self.kernel_name = kernel_name
        self.dynamic = kernel_meta.get('dynamic')
        self.device_id = kernel_meta.get('device_index')
        self.arch = kernel_meta.get('device_name')
        self.output_so_dir = os.getenv("KERNEL_META_DIR", default="akg_kernel_meta")
        self.torch_path = str(torch_path)
        os.makedirs(self.output_so_dir, exist_ok=True)
        backend = kernel_meta.get('backend')
        self.backend = backend if backend is not None else "ascend"
        self.block_dim = 1

    def _write_mlir(self, input_mlir):
        mlir_path = os.path.join(self.output_so_dir, f"{self.kernel_name}.mlir")
        with open(mlir_path, "w", encoding="utf-8") as f:
            f.write(input_mlir)
        return mlir_path

    def get_launcher(self, lib_path=None):
        """ get device kernel launcher. """
        if self.dynamic:
            if lib_path is None:
                lib_path = os.path.join(self.output_so_dir, f"lib{self.kernel_name}.so")
            kernel_func, tiling_func, _, tiling_size = get_host_functions(self.kernel_name, lib_path)
        else:
            if lib_path is None:
                lib_path = os.path.join(self.output_so_dir, f"{self.kernel_name}.o")
            kernel_func = get_device_function(self.kernel_name, lib_path)
            tiling_func = 0
            tiling_size = 0
        def kernel_call(*args, stream=None):
            torch_launch(self.kernel_name, self.torch_path, kernel_func, tiling_func, tiling_size,
                         self.block_dim, stream, self.dynamic, *args)

        return kernel_call


    def compile(self, input_mlir: str):
        """ Compile .mlir file to .so file. """
        self._write_mlir(input_mlir)

        input_file = os.path.join(self.output_so_dir, self.kernel_name + ".mlir")
        out_file = os.path.join(self.output_so_dir, self.kernel_name + "_out.mlir")

        # get akg_tools_dir
        akg_tools_dir = os.path.dirname(os.path.abspath(__file__))

        # run akg-opt
        run_akg_opt(
            input_file=input_file,
            output_file=out_file,
            akg_tools_dir=akg_tools_dir,
            dyn_shape=self.dynamic,
            enable_loop_fusion=True,
            arch=self.arch,
            mlir_timing=True
        )

        self.block_dim = get_block_dim_from_mlir(out_file)

        output_so_path = os.path.join(self.output_so_dir, f"{self.kernel_name}.so")
        dump_log = os.path.join(self.output_so_dir, self.kernel_name + "_dump_bishengir.log")
        try:
            ascend_compile(out_file, output_so_path, block_dim=self.block_dim, dump_log_path=dump_log)
            logging.debug("compile finish, lib.so save to %s", os.path.abspath(output_so_path))
            dump_ascend_meta_data(self.output_so_dir, self.kernel_name, block_dim=self.block_dim)
        except Exception as compile_err:
            raise RuntimeError(
                f"compile MLIR failed, error message: {str(compile_err)}"
            ) from compile_err

        self.launcher = self.get_launcher()

    def run(self, *args, **kwargs):
        """ launch .so file by akg_ascend_backend. """

        try:
            self.launcher(*args, **kwargs)
            logging.debug("success launch kernel: %s", {self.kernel_name})
        except Exception as running_err:
            raise RuntimeError(
                f"exec {self.kernel_name}.so error, error msg: {str(running_err)}"
            ) from running_err
