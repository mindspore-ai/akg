#!/usr/bin/env python3
# encoding: utf-8
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
# ============================================================================
"""setup package for ms_custom_ops."""

import logging
import os
import sys
import shutil
import multiprocessing
from typing import List
from pathlib import Path
from setuptools import find_packages, setup
from setuptools.command.build_ext import build_ext
from setuptools import Extension
import subprocess


ROOT_DIR = os.path.dirname(__file__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

package_name = "ms_custom_ops"

if not sys.platform.startswith("linux"):
    logger.warning(
        "ms_custom_ops only supports Linux platform."
        "Building on %s, "
        "so ms_custom_ops may not be able to run correctly",
        sys.platform,
    )


def get_path(*filepath) -> str:
    return os.path.join(ROOT_DIR, *filepath)


def read_readme() -> str:
    """Read the README file if present."""
    p = get_path("README.md")
    if os.path.isfile(p):
        with open(get_path("README.md"), encoding="utf-8") as f:
            return f.read()
    else:
        return ""


def get_requirements() -> List[str]:
    """Get Python package dependencies from requirements.txt."""

    def _read_requirements(filename: str) -> List[str]:
        requirements_path = get_path(filename)
        if not os.path.exists(requirements_path):
            return []
        
        with open(requirements_path) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif line.startswith("--"):
                continue
            elif "http" in line:
                continue
            elif line.strip() == "":
                continue
            else:
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("requirements.txt")
    return requirements


def write_commit_id():
    ret_code = os.system("git rev-parse --abbrev-ref HEAD > .commit_id "
                         "&& git log --abbrev-commit -1 >> .commit_id")
    if ret_code != 0:
        sys.stdout.write("Warning: Can not get commit id information. Please make sure git is available.")
        os.system("echo 'git is not available while building.' > .commit_id")


def get_version():
    """Get version from version.txt or use default."""
    version_path = Path("ms_custom_ops") / "version.txt"
    if version_path.exists():
        return version_path.read_text().strip()
    else:
        return "0.1.0"

version = get_version()

def _get_ascend_home_path():
    return os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")

def _get_ascend_env_path():
    env_script_path = os.path.realpath(os.path.join(_get_ascend_home_path(), "..", "set_env.sh"))
    if not os.path.exists(env_script_path):
        raise ValueError(f"The file '{env_script_path}' is not found, "
                            "please make sure environment variable 'ASCEND_HOME_PATH' is set correctly.")
    return env_script_path

class CustomBuildExt(build_ext):
    ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

    def build_extension(self, ext):
        if ext.name == "ms_custom_ops":
            self.build_ms_custom_ops(ext)
        else:
            raise ValueError(f"Unknown extension name: {ext.name}")

    def build_ms_custom_ops(self, ext):
        ext_name = ext.name
        so_name = ext_name + ".so"
        logger.info(f"Building {so_name} ...")
        OPS_DIR = os.path.join(ROOT_DIR, "ccsrc")
        BUILD_OPS_DIR = os.path.join(ROOT_DIR, "build", "ms_custom_ops")
        os.makedirs(BUILD_OPS_DIR, exist_ok=True)

        ascend_home_path = _get_ascend_home_path()
        env_script_path = _get_ascend_env_path()
        build_extension_dir = os.path.join(BUILD_OPS_DIR, "kernel_meta", ext_name)
        dst_so_path = self.get_ext_fullpath(ext.name)
        dst_dir = os.path.dirname(dst_so_path)
        package_path = os.path.join(dst_dir, package_name)
        os.makedirs(package_path, exist_ok=True)
        
        # Also prepare the Python package directory for generated files
        python_package_path = os.path.join(ROOT_DIR, "python", package_name)
        os.makedirs(python_package_path, exist_ok=True)
        # 动态检测CPU核心数，取一半，至少为1
        available_cores = multiprocessing.cpu_count()
        compile_cores = max(1, available_cores // 2)
        logger.info(f"Available CPU cores: {available_cores}, using {compile_cores} cores for compilation")
        # Combine all cmake commands into one string
        cmake_cmd = (
            f"source {env_script_path} && "
            f"cmake -S {OPS_DIR} -B {BUILD_OPS_DIR}"
            f"  -DCMAKE_BUILD_TYPE=Release"
            f"  -DCMAKE_INSTALL_PREFIX={os.path.join(BUILD_OPS_DIR, 'install')}"
            f"  -DBUILD_EXTENSION_DIR={build_extension_dir}"
            f"  -DASCENDC_INSTALL_PATH={package_path}"
            f"  -DMS_EXTENSION_NAME={ext_name}"
            f"  -DASCEND_CANN_PACKAGE_PATH={ascend_home_path} && "
            f"cmake --build {BUILD_OPS_DIR} -j{compile_cores} --verbose"
        )

        try:
            # Run the combined cmake command
            logger.info(f"Running combined CMake commands:\n{cmake_cmd}")
            result = subprocess.run(cmake_cmd, cwd=self.ROOT_DIR, text=True, shell=True, capture_output=False)
            if result.returncode != 0:
                logger.info("CMake commands failed:")
                logger.info(result.stdout)  # Print standard output
                logger.info(result.stderr)  # Print error output
                raise RuntimeError(f"Combined CMake commands failed with exit code {result.returncode}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to build {so_name}: {e}")

        # Copy the generated .so file to the target directory
        src_so_path = os.path.join(build_extension_dir, so_name)
        if os.path.exists(dst_so_path):
            os.remove(dst_so_path)
        so_name = os.path.basename(dst_so_path)
        shutil.copy(src_so_path, os.path.join(package_path, so_name))
        logger.info(f"Copied {so_name} to {dst_so_path}")

        # Copy generated Python files to Python package directory
        auto_generate_dir = os.path.join(build_extension_dir, "auto_generate")
        if os.path.exists(auto_generate_dir):
            generated_files = ["gen_ops_def.py", "gen_ops_prim.py"]
            for gen_file in generated_files:
                src_gen_path = os.path.join(auto_generate_dir, gen_file)
                if os.path.exists(src_gen_path):
                    dst_gen_path = os.path.join(python_package_path, gen_file)
                    shutil.copy(src_gen_path, dst_gen_path)
                    replace_cmd = ["sed", "-i", "s/import ms_cusrom_ops/from . import ms_custom_ops/g", dst_gen_path]
                    try:
                        result = subprocess.run(replace_cmd, cwd=self.ROOT_DIR, text=True, shell=False)
                    except subprocess.CalledProcessError as e:
                        raise RuntimeError(f"Failed to exec command {replace_cmd}: {e}")
                    logger.info(f"Copied {gen_file} to {dst_gen_path}")
                else:
                    logger.warning(f"Generated file not found: {src_gen_path}")
        else:
            logger.warning(f"Auto-generate directory not found: {auto_generate_dir}")




write_commit_id()

package_data = {
    "": [
        "*.so",
        "lib/*.so",
        ".commit_id"
    ],
    "ms_custom_ops": [
        "gen_ops_def.py",
        "gen_ops_prim.py"
    ]
}

def _get_ext_modules():
    ext_modules = []
    if os.path.exists(_get_ascend_home_path()):
        # sources are specified in CMakeLists.txt
        ext_modules.append(Extension("ms_custom_ops", sources=[]))
    return ext_modules

setup(
    name=package_name,
    version=version,
    author="MindSpore Team",
    license="Apache 2.0",
    description=(
        "MindSpore Custom Operations for Ascend NPU"
    ),
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://gitee.com/mindspore/ms-custom-ops",
    project_urls={
        "Homepage": "https://gitee.com/mindspore/ms-custom-ops",
        "Documentation": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    packages=find_packages(where="python"),
    package_dir={"": "python"},
    python_requires=">=3.9",
    install_requires=get_requirements(),
    cmdclass={"build_ext": CustomBuildExt},
    ext_modules=_get_ext_modules(),
    include_package_data=True,
    package_data=package_data,
)
