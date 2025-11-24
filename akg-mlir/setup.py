import os
import sys
import glob
import shutil
import subprocess
import multiprocessing

from distutils.command.build import build as _build
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py


def _check_env_flag(name: str, default=None) -> bool:
    return str(os.getenv(name, default)).upper() in ["ON", "1", "YES", "TRUE", "Y"]

def _read_version():
    try:
        with open("version.txt", "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "2.0.0"


BASE_DIR = os.path.dirname(os.path.realpath(__file__))

# If true, enable LTC build by default
AKG_ENABLE_LTC = _check_env_flag("AKG_ENABLE_LTC", True)
AKG_ENABLE_BINDINGS_PYTHON = _check_env_flag("AKG_ENABLE_BINDINGS_PYTHON", False)
LLVM_INSTALL_DIR = os.getenv("LLVM_INSTALL_DIR", None)
CMAKE_BUILD_TYPE = os.getenv("CMAKE_BUILD_TYPE", "Release")

AKG_CMAKE_ALREADY_BUILD = _check_env_flag( "AKG_CMAKE_ALREADY_BUILD", False)
AKG_CMAKE_BUILD_DIR = os.getenv("AKG_CMAKE_BUILD_DIR")
MAX_JOBS = os.getenv("MAX_JOBS", str(multiprocessing.cpu_count()))


class AkgBuild(_build):
    def initialize_options(self):
        _build.initialize_options(self)
        self.build_base = "build"

    def run(self):
        self.run_command("build_ext")
        self.run_command("build_py")
        self.run_command("build_scripts")


class CMakeBuild(build_ext):
    def copy_so(self, cmake_build_dir, target_dir):
        dst = os.path.join(target_dir, "akg")
        os.makedirs(dst, exist_ok=True)
        generated_so_files = glob.glob(os.path.join(cmake_build_dir, "lib", '*.so'), recursive=True)
        for src in generated_so_files:
            shutil.copy2(src, dst, follow_symlinks=False)

    def cmake_build(self, cmake_build_dir):
        cmake_config_args = [
            f"cmake",
            f"-DCMAKE_BUILD_TYPE={CMAKE_BUILD_TYPE}",
            f"{BASE_DIR}",
        ]
        if AKG_ENABLE_BINDINGS_PYTHON:
            cmake_config_args = [ f"-DAKG_ENABLE_BINDINGS_PYTHON=ON", ]

        if LLVM_INSTALL_DIR:
            cmake_config_args += [
                f"-DMLIR_DIR='{LLVM_INSTALL_DIR}/lib/cmake/mlir/'",
                f"-DLLVM_DIR='{LLVM_INSTALL_DIR}/lib/cmake/llvm/'",
            ]

        cmake_build_args = [
            f"cmake",
            f"--build",
            f".",
            f"--config",
            f"{CMAKE_BUILD_TYPE}",
            f"--",
            f"-j{MAX_JOBS}",
        ]
        try:
            subprocess.check_call(cmake_config_args, cwd=cmake_build_dir)
            subprocess.check_call(cmake_build_args, cwd=cmake_build_dir)
        except subprocess.CalledProcessError as e:
            print("cmake build failed with\n", e)
            print("debug by follow cmake command:")
            sys.exit(e.returncode)
        finally:
            print(f"cmake config: {' '.join(cmake_config_args)}")
            print(f"cmake build: {' '.join(cmake_build_args)}")
            print(f"cmake workspace: {cmake_build_dir}")

    def run(self):
        target_dir = self.build_lib
        cmake_build_dir = AKG_CMAKE_BUILD_DIR
        if not cmake_build_dir:
            cmake_build_dir = os.path.abspath(
                os.path.join(target_dir, "..")
            )
        python_package_dir = os.path.join(cmake_build_dir, "python_packages")
        if not AKG_CMAKE_ALREADY_BUILD:
            os.makedirs(cmake_build_dir, exist_ok=True)
            mlir_libs_dir = os.path.join(python_package_dir,  "akg", "akg_mlir", "_mlir_libs")
            if os.path.exists(mlir_libs_dir):
                print(f"Removing _mlir_mlibs dir to force rebuild: {mlir_libs_dir}")
                shutil.rmtree(mlir_libs_dir)
            else:
                print(f"Not removing _mlir_libs dir (does not exist): {mlir_libs_dir}")
            self.cmake_build(cmake_build_dir)

        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=False, onerror=None)

        if AKG_ENABLE_BINDINGS_PYTHON:
            shutil.copytree(python_package_dir, target_dir, symlinks=False)

        self.copy_so(cmake_build_dir, target_dir)

        bin_src = os.path.join(cmake_build_dir, "bin")
        bin_dst = os.path.join(target_dir, "akg", "bin")
        shutil.copytree(bin_src, bin_dst, symlinks=False)


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class PythonPackageBuild(build_py, object):
    def get_src_py_and_dst(self):
        target_dir = self.build_lib
        ret = []
        generated_python_files = glob.glob(os.path.join(BASE_DIR, "python/akg_mlir", '**/*.py'), recursive=True)
        for src in generated_python_files:
            dst = os.path.join(os.path.join(target_dir, "akg"),
                               os.path.relpath(src, os.path.join(BASE_DIR, "python/akg_mlir")))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            ret.append((src, dst))
        return ret
    def run(self) -> None:
        ret = self.get_src_py_and_dst()
        for src, dst in ret:
            self.copy_file(src, dst)
        super(PythonPackageBuild, self).finalize_options()

# 读取 README.md 作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


INSTALL_REQUIRES = [
    "numpy",
]
EXT_MODULES = [
    CMakeExtension("akg_mlir._mlir_libs._akgMlir"),
]

NAME = "akg"

setup(
    name=NAME,
    version=_read_version(),
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
            "akg_benchmark = akg.exec_tools.py_benchmark:main",
        ],
    },
    zip_safe=False,
)
