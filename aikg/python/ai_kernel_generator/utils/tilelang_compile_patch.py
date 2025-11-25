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

import subprocess
from pathlib import Path


def patch_tilelang_compiler():
    """动态补丁 tilelang 编译器，添加详细的错误信息显示功能"""
    try:
        from tilelang.jit import jit_npu
    except ImportError:
        return False

    if not hasattr(jit_npu, 'compiler_npu'):
        return False

    compiler_class = jit_npu.compiler_npu

    # 检查是否已经被补丁过了
    if hasattr(compiler_class._npuir_to_bin_enable_npu_compile, '_aikg_patched'):
        return True

    original_compile_method = compiler_class._npuir_to_bin_enable_npu_compile

    def patched_npuir_to_bin_enable_npu_compile(self):
        """补丁后的编译方法,提供详细的错误信息"""
        import tempfile
        import os
        from pathlib import Path

        linalg = self.mlir_content
        metadata = self.metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            ttadapter_path = os.path.join(tmpdir, "kernel.npuir")
            Path(ttadapter_path).write_text(linalg)
            bin_file = os.path.join(tmpdir, "kernel")
            bin_path = os.path.join(tmpdir, "kernel.o")

            npu_compiler_path = jit_npu._get_npucompiler_path()
            _compile_option_list = [
                "--enable-auto-multi-buffer=true", "--enable-triton-kernel-compile=true",
                "--enable-hivm-compile=true", "--disable-hivm-tensor-compile=true"
            ]
            cmd_list = ([npu_compiler_path, ttadapter_path] + _compile_option_list +
                        ["-o", bin_file])

            try:
                ret = subprocess.run(cmd_list, capture_output=True, check=True, text=True)
            except subprocess.CalledProcessError as e:
                # 显示完整的编译错误信息
                error_msg = f"\n{'='*60}\n"
                error_msg += f"NPU Compiler Error (exit code {e.returncode})\n"
                error_msg += f"Command: {' '.join(cmd_list)}\n"
                if e.stdout:
                    error_msg += f"\nSTDOUT:\n{e.stdout}\n"
                if e.stderr:
                    error_msg += f"\nSTDERR:\n{e.stderr}\n"
                raise RuntimeError(error_msg) from e

            return Path(bin_path).read_bytes()

    # 应用补丁
    try:
        compiler_class._npuir_to_bin_enable_npu_compile = patched_npuir_to_bin_enable_npu_compile
        # 标记已经被补丁过了
        compiler_class._npuir_to_bin_enable_npu_compile._aikg_patched = True
        return True
    except (AttributeError, TypeError) as e:
        print(f"Warning: Failed to patch TileLang compiler: {e}")
        return False


def apply_tilelang_patches():
    """应用所有 tilelang 补丁"""
    success = patch_tilelang_compiler()
    return success


# 自动应用补丁(当模块被导入时)
if __name__ != "__main__":
    apply_tilelang_patches()

# 测试代码
if __name__ == "__main__":
    print("Testing TileLang patches...")
    success = patch_tilelang_compiler()

    if success:
        print("✓ TileLang compiler patch applied successfully!")
    else:
        print("✗ Failed to apply TileLang compiler patch (tilelang may not be installed)")
