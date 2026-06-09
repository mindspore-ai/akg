# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
"""AKG MLIR setup module."""
import os
import shutil
import subprocess
import multiprocessing
import logging
from pathlib import Path
from typing import List, Tuple

from setuptools import setup, Extension
from setuptools.command.build import build
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_env_flag(name: str, default: str = "OFF") -> bool:
    """check env flag"""
    val = os.getenv(name, default)
    return val.upper() in ("ON", "1", "YES", "TRUE", "Y")


def read_version() -> str:
    """Read the package version from version.txt."""
    try:
        version_file = Path(__file__).parent / "version.txt"
        return version_file.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return "3.0.0"

def get_cmake():
    """get cmake"""
    return shutil.which("cmake")

def get_cmake_generator() -> str:
    """Determine the CMake generator to use."""
    if shutil.which("ninja"):
        logger.info("Using Ninja generator for faster builds")
        return "Ninja"
    return "Unix Makefiles"


def check_cmake_version(min_version: str = "3.15") -> None:
    """get cmake version"""
    try:
        result = subprocess.run(
            [get_cmake(), "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        version_str = result.stdout.split()[2]
        logger.info("CMake version: %s", version_str)
    except Exception as e:
        raise RuntimeError("CMake not found. Please install CMake >= 3.15") from e


BASE_DIR = Path(__file__).parent.resolve()

AKG_ENABLE_LTC = check_env_flag("AKG_ENABLE_LTC", "ON")
AKG_ENABLE_BINDINGS_PYTHON = check_env_flag("AKG_ENABLE_BINDINGS_PYTHON", "OFF")
LLVM_INSTALL_DIR = os.getenv("LLVM_INSTALL_DIR")
CMAKE_BUILD_TYPE = os.getenv("CMAKE_BUILD_TYPE", "Release")
AKG_CMAKE_ALREADY_BUILD = check_env_flag("AKG_CMAKE_ALREADY_BUILD", "OFF")
AKG_CMAKE_BUILD_DIR = os.getenv("AKG_CMAKE_BUILD_DIR")
MAX_JOBS = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))


class CMakeConfig:
    """CMake config"""
    def __init__(self):
        self.build_type = CMAKE_BUILD_TYPE
        self.generator = get_cmake_generator()
        self.max_jobs = MAX_JOBS
        self.base_dir = BASE_DIR

    def get_configure_args(self) -> List[str]:
        """get cmake args"""
        args = [
            get_cmake(),
            "-G", self.generator,
            f"-DCMAKE_BUILD_TYPE={self.build_type}",
            str(self.base_dir),
        ]

        if AKG_ENABLE_BINDINGS_PYTHON:
            args.append("-DAKG_ENABLE_BINDINGS_PYTHON=ON")

        if LLVM_INSTALL_DIR:
            llvm_path = Path(LLVM_INSTALL_DIR)
            args.extend([
                f"-DMLIR_DIR={llvm_path / 'lib' / 'cmake' / 'mlir'}",
                f"-DLLVM_DIR={llvm_path / 'lib' / 'cmake' / 'llvm'}",
            ])

        return args

    def get_build_args(self) -> List[str]:
        """get build args"""
        return [
            get_cmake(),
            "--build", ".",
            "--config", self.build_type,
            "--", f"-j{self.max_jobs}", ]


class AkgBuild(build):
    """AKG build config"""
    def initialize_options(self):
        super().initialize_options()
        self.build_base = "build"

    def run(self):
        logger.info("Starting AKG build process")
        self.run_command("build_ext")
        self.run_command("build_py")
        self.run_command("build_scripts")


class CMakeBuild(build_ext):
    """AKG cmake build"""
    def copy_so_files(self, cmake_build_dir: Path, target_dir: Path) -> None:
        """copy dynamic libraries to dest"""
        so_files = list(cmake_build_dir.glob("lib/**/*.so"))

        lib_so_files = [f for f in so_files if f.name.startswith('lib')]
        so_files = [f for f in so_files if not f.name.startswith('lib')]

        lib_dir = target_dir / "akg" / "lib"
        lib_dir.mkdir(parents=True, exist_ok=True)

        akg_dir = target_dir / "akg"
        akg_dir.mkdir(parents=True, exist_ok=True)

        if lib_so_files:
            for src_file in lib_so_files:
                shutil.copy2(src_file, lib_dir, follow_symlinks=False)

        if so_files:
            for src_file in so_files:
                shutil.copy2(src_file, akg_dir, follow_symlinks=False)

    def cmake_build(self, cmake_build_dir: Path) -> None:
        """cmake build"""
        cmake_config = CMakeConfig()
        configure_args = cmake_config.get_configure_args()
        build_args = cmake_config.get_build_args()

        logger.info("CMake configure: %s", configure_args)
        logger.info("CMake build: %s", build_args)
        logger.info("CMake workspace: %s", cmake_build_dir)

        try:
            subprocess.check_call(configure_args, cwd=cmake_build_dir)
            subprocess.check_call(build_args, cwd=cmake_build_dir)
            logger.info("CMake build completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error("CMake build failed!")
            logger.error("CMake configure: %s", configure_args)
            logger.error("CMake build: %s", build_args)
            logger.error("CMake workspace: %s", cmake_build_dir)
            if e.output:
                logger.error("Error output: %s", e.output.decode() if isinstance(e.output, bytes) else e.output)
            raise RuntimeError(f"CMake build failed with return code {e.returncode}") from e

    def verify_build(self, target_dir: Path) -> None:
        """verify build target"""
        expected_files = [
            target_dir / "akg" / "bin",
            target_dir / "akg" / "lib",
        ]
        if AKG_ENABLE_BINDINGS_PYTHON:
            expected_files.append(target_dir / "akg" / "_mlir_libs")

        for file_path in expected_files:
            if not file_path.exists():
                raise RuntimeError(f"Build verification failed: Missing {file_path}")

        logger.info("Build verification passed")

    def run(self):
        """run build"""
        target_dir = Path(self.build_lib)
        cmake_build_dir = Path(AKG_CMAKE_BUILD_DIR) if AKG_CMAKE_BUILD_DIR else target_dir.parent
        python_package_dir = cmake_build_dir / "python_packages"

        if not AKG_CMAKE_ALREADY_BUILD:
            cmake_build_dir.mkdir(parents=True, exist_ok=True)

            mlir_libs_dir = python_package_dir / "akg" / "akg_mlir" / "_mlir_libs"
            if mlir_libs_dir.exists():
                logger.info("Removing _mlir_libs dir to force rebuild: %s", mlir_libs_dir)
                shutil.rmtree(mlir_libs_dir)
            else:
                logger.info("_mlir_libs dir does not exist: %s", mlir_libs_dir)

            check_cmake_version()
            self.cmake_build(cmake_build_dir)

        if target_dir.exists():
            logger.info("Cleaning target directory: %s", target_dir)
            shutil.rmtree(target_dir)

        if AKG_ENABLE_BINDINGS_PYTHON:
            logger.info("Copying Python packages from %s to %s", python_package_dir, target_dir)
            shutil.copytree(python_package_dir, target_dir, symlinks=False)

        self.copy_so_files(cmake_build_dir, target_dir)

        bin_src = cmake_build_dir / "bin"
        bin_dst = target_dir / "akg" / "bin"
        if bin_src.exists():
            shutil.copytree(bin_src, bin_dst, symlinks=False)

        self.verify_build(target_dir)


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = ""):
        super().__init__(name, sources=[])
        self.sourcedir = Path(sourcedir).resolve() if sourcedir else BASE_DIR


class PythonPackageBuild(build_py):
    """python build"""
    def get_src_py_and_dst(self) -> List[Tuple[Path, Path]]:
        """init src dir and dst dir"""
        target_dir = Path(self.build_lib)
        src_dir = BASE_DIR / "python" / "akg_mlir"

        py_files = list(src_dir.glob("**/*.py"))

        result = []
        for src_file in py_files:
            rel_path = src_file.relative_to(src_dir)
            dst_file = target_dir / "akg" / rel_path
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            result.append((src_file, dst_file))

        return result

    def run(self) -> None:
        logger.info("Building Python package")
        src_dst_pairs = self.get_src_py_and_dst()

        for src, dst in src_dst_pairs:
            self.copy_file(str(src), str(dst))

        super().finalize_options()
        logger.info("Python package build completed")


try:
    readme_path = BASE_DIR / "README.md"
    long_description = readme_path.read_text(encoding="utf-8")
except FileNotFoundError:
    logger.warning("README.md not found, using default description")
    long_description = "AKG MLIR - An optimizer for operators in Deep Learning Networks"


INSTALL_REQUIRES = [
    "numpy",
]

EXT_MODULES = [
    CMakeExtension("akg_mlir._mlir_libs._akgMlir"),
]

setup(
    name="akg",
    version=read_version(),
    author="The MindSpore Authors",
    author_email="contact@mindspore.cn",
    description="An optimizer for operators in Deep Learning Networks, which provides the ability to automatically "
                "fuse ops with specific patterns.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://www.mindspore.cn/",
    download_url="https://gitee.com/mindspore/akg/tags",
    project_urls={
        'Sources': 'https://gitee.com/mindspore/akg',
        'Issue Tracker': 'https://gitee.com/mindspore/akg/issues',
    },
    license="Apache 2.0",
    include_package_data=True,
    cmdclass={
        "build": AkgBuild,
        "build_ext": CMakeBuild,
        "build_py": PythonPackageBuild,
    },
    ext_modules=EXT_MODULES,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    entry_points={
        "console_scripts": [
            "akg_benchmark = akg.exec_tools.benchmark:main",
        ],
    },
    zip_safe=False,
)
