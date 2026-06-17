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

"""Setup script for MFUSE MLIR Python package."""

import filecmp
import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

PACKAGE_VERSION = "1.0"


def _get_git_commit_id(repo_root: Path) -> str:
    """Resolve the current git commit id for packaging metadata."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _write_commit_id_file(target_dir: Path, repo_root: Path) -> None:
    """Write the git commit id into the installed Python package."""
    commit_id = _get_git_commit_id(repo_root)
    (target_dir / ".commit_id").write_text(f"{commit_id}\n", encoding="utf-8")


class CMakeBuild(build_ext):
    """Custom build command to compile C++ extensions."""

    @staticmethod
    def _is_enabled(env_name, default):
        """Return whether an on/off style environment variable is enabled."""
        return os.environ.get(env_name, default).upper() not in ("0", "OFF", "FALSE", "NO")

    @staticmethod
    def _is_elf(path):
        if path.is_symlink() or not path.is_file():
            return False
        with path.open("rb") as f:
            return f.read(4) == b"\x7fELF"

    @staticmethod
    def _find_strip_tool():
        for env_name in ("CMAKE_STRIP", "STRIP"):
            strip_tool = os.environ.get(env_name)
            if strip_tool:
                return strip_tool
        cross_compile = os.environ.get("CROSS_COMPILE", "")
        return shutil.which(f"{cross_compile}strip")

    @staticmethod
    def _remove_duplicate_capi_library(mlir_libs_dir):
        """Remove the unversioned aggregate CAPI library when it duplicates the versioned one."""
        lib_name = "libMFusionAggregateCAPI.so"
        unversioned_lib = mlir_libs_dir / lib_name
        versioned_libs = sorted(mlir_libs_dir.glob(f"{lib_name}.*"))
        if not unversioned_lib.exists() or not versioned_libs:
            return

        if unversioned_lib.is_symlink():
            unversioned_lib.unlink()
            print(f"Removed duplicate CAPI library: {unversioned_lib}")
            return

        duplicate_lib = next(
            (
                versioned_lib
                for versioned_lib in versioned_libs
                if filecmp.cmp(unversioned_lib, versioned_lib, shallow=False)
            ),
            None,
        )
        if duplicate_lib:
            unversioned_lib.unlink()
            print(f"Removed duplicate CAPI library: {unversioned_lib}")
            return

        versioned_lib_names = ", ".join(versioned_lib.name for versioned_lib in versioned_libs)
        print(
            "Warning: keeping libMFusionAggregateCAPI.so because it differs from "
            f"versioned libraries: {versioned_lib_names}"
        )

    def _strip_release_binaries(self, mlir_libs_dir, build_type):
        """Strip ELF binaries in release builds to reduce wheel size."""
        if build_type.lower() != "release" or not self._is_enabled("MFUSION_STRIP", "ON"):
            return

        strip_tool = self._find_strip_tool()
        if not strip_tool:
            print("Warning: strip tool not found; skipping release binary stripping")
            return

        for binary in sorted(mlir_libs_dir.rglob("*")):
            if not self._is_elf(binary):
                continue
            before_size = binary.stat().st_size
            subprocess.check_call([strip_tool, "--strip-unneeded", str(binary)])
            after_size = binary.stat().st_size
            print(f"Stripped {binary}: {before_size} -> {after_size} bytes")

    def run(self):
        """Build C++ extensions using CMake."""
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        cmake_args = []

        # Get BUILD_TYPE from environment variable, default to Release
        build_type = os.environ.get("BUILD_TYPE", "Release")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")
        cmake_prefix_path = os.environ.get("CMAKE_PREFIX_PATH")
        if cmake_prefix_path:
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={cmake_prefix_path}")

        enable_asan = os.environ.get("ENABLE_ASAN", "OFF")
        cmake_args.append(f"-DENABLE_ASAN={enable_asan}")

        # Configure with CMake
        if os.environ.get("INC_BUILD", "0") != "1":
            subprocess.check_call(
                [
                    "cmake",
                    "-S",
                    str(Path(__file__).parent),
                    "-B",
                    self.build_temp,
                ]
                + cmake_args
            )

        # Build with CMake
        build_jobs = os.environ.get("BUILD_JOBS", "8")
        build_tests = os.environ.get("BUILD_TESTS", "OFF")
        subprocess.check_call(["cmake", "--build", self.build_temp, "-j", build_jobs])

        # Copy the generated mfusion package
        python_package_dir = Path(self.build_temp) / "python_packages" / "mfusion"
        target_dir = Path(self.build_lib) / "mfusion"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(python_package_dir, target_dir)

        # Merge Python sources from the repo.
        source_python_dir = Path(__file__).parent / "python" / "mfusion"
        if source_python_dir.exists():
            shutil.copytree(source_python_dir, target_dir, dirs_exist_ok=True)
        _write_commit_id_file(target_dir, Path(__file__).parent)

        # Copy tests
        if build_tests == "ON":
            tests_dir = Path(self.build_temp) / "tests"
            if tests_dir.exists():
                build_dir = Path(__file__).parent / "build"
                if not build_dir.exists():
                    raise RuntimeError(f"Directory does not exist: {build_dir}. Please run build.sh to create it.")
                dst_tests = build_dir / "tests"
                if dst_tests.exists():
                    shutil.rmtree(dst_tests)
                shutil.copytree(tests_dir, dst_tests)

        # Copy mfusion-opt executable to _mlir_libs
        mfusion_opt_src = Path(self.build_temp) / "bin" / "mfusion-opt"
        mfusion_opt_dst = target_dir / "_mlir_libs" / "mfusion-opt"
        if mfusion_opt_src.exists():
            shutil.copy2(mfusion_opt_src, mfusion_opt_dst, follow_symlinks=False)
            print(f"Copied mfusion-opt to {mfusion_opt_dst}")
        else:
            print(f"Warning: mfusion-opt not found at {mfusion_opt_src}")

        mlir_libs_dir = target_dir / "_mlir_libs"
        self._remove_duplicate_capi_library(mlir_libs_dir)
        self._strip_release_binaries(mlir_libs_dir, build_type)


class BuildPyWithExt(build_py):
    def run(self):
        super().run()
        self.run_command("build_ext")


setup(
    cmdclass={
        "build_ext": CMakeBuild,
        "build_py": BuildPyWithExt,
    }
)
