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

"""Setup script for MUSE MLIR Python package."""

import os
import shutil
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


class CMakeBuild(build_ext):
    """Custom build command to compile C++ extensions."""

    def run(self):
        """Build C++ extensions using CMake."""
        Path(self.build_temp).mkdir(parents=True, exist_ok=True)

        cmake_args = []

        # Get BUILD_TYPE from environment variable, default to Release
        build_type = os.environ.get("BUILD_TYPE", "Release")
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_type}")

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
        subprocess.check_call(["cmake", "--build", self.build_temp, "-j", build_jobs])

        # Copy the generated mfusion package
        python_package_dir = Path(self.build_temp) / "python_packages" / "mfusion"
        target_dir = Path(self.build_lib) / "mfusion"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        shutil.copytree(python_package_dir, target_dir)


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
