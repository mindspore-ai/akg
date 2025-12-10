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
import re
import logging

from akg.backends.ascend import transform_data_to_ascend, launch, ascend_compile

logging.basicConfig(level=logging.INFO)


class Kernel:
    """ Kernel for support ascend_npu_ir """
    def __init__(self, kernel_meta=None):
        self.kernel_name = kernel_meta.get('kernel_name')
        self.dynamic = kernel_meta.get('dynamic')
        self.device_id = kernel_meta.get('device_index')
        self.output_so_dir = os.getenv("KERNEL_META_DIR", default="akg_kernel_meta")
        os.makedirs(self.output_so_dir, exist_ok=True)
        backend = kernel_meta.get('backend')
        self.backend = backend if backend is not None else "ascend"
        num_outputs = kernel_meta.get('num_outputs')
        self.output_indexes = self._get_output_index(num_outputs)

    def _get_output_index(self, num_outputs: int):
        return [-i for i in range(1, num_outputs + 1)]

    def _write_mlir(self, input_mlir):
        mlir_path = os.path.join(self.output_so_dir, f"{self.kernel_name}_out.mlir")
        if not self.dynamic:
            pattern = r'(\{[^{}]*\{[^{}]*)<[^<>]*>'
            replacement = r'\1<HOST>'
            input_mlir = re.sub(pattern, replacement, input_mlir, count=1)
        with open(mlir_path, "w", encoding="utf-8") as f:
            f.write(input_mlir)
        return mlir_path

    def compile(self, input_mlir: str):
        """ Compile .mlir file to .so file. """
        mlir_file_path = self._write_mlir(input_mlir)
        output_so_path = os.path.join(self.output_so_dir, f"{self.kernel_name}.so")
        try:
            ascend_compile(mlir_file_path, output_so_path)
            logging.info("compile finish, lib.so save to %s", os.path.abspath(output_so_path))
        except Exception as compile_err:
            raise Exception(
                f"compile MLIR failed, error message: {str(compile_err)}"
            ) from compile_err

    def run(self, *args, **kwargs):
        """ launch .so file by akg_ascend_backend """
        n = len(args)
        try:
            input_for_mod_ctypes = transform_data_to_ascend(
                args[:n-1],
                self.kernel_name,
                self.output_indexes,
                self.dynamic,
                "ascend"
            )
            launch(
                self.output_so_dir,
                self.kernel_name,
                self.device_id,
                self.dynamic,
                *input_for_mod_ctypes
            )
            logging.info("success launch kernel: %s", {self.kernel_name})
        except Exception as running_err:
            raise Exception(
                f"exec {self.kernel_name}.so error, error msg: {str(running_err)}"
            ) from running_err
