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
"""setup package for custom compiler tool"""
import argparse
import json
import os
import re
import subprocess
import shutil
import tempfile
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OP_HOST = "op_host"
OP_KERNEL = "op_kernel"
code_suffix = {"cpp", "h"}


SOC_VERSION_MAP = {
    "ascend910a": "ascend910",
    "ascend910proa": "ascend910",
    "ascned910premiuma": "ascend910",
    "ascend910prob": "ascend910",
    "ascend910b": "ascend910b",
    "ascend910b1": "ascend910b",
    "ascend910b2": "ascend910b",
    "ascend910b2c": "ascend910b",
    "ascend910b3": "ascend910b",
    "ascend910b4": "ascend910b",
    "ascend910b4-1": "ascend910b",
    "ascend910c": "ascend910_93",
    "ascend910_9391": "ascend910_93",
    "ascend910_9392": "ascend910_93",
    "ascend910_9381": "ascend910_93",
    "ascend910_9382": "ascend910_93",
    "ascend910_9372": "ascend910_93",
    "ascend910_9362": "ascend910_93",
    "ascend310p": "ascend310p",
    "ascend310p1": "ascend310p",
    "ascend310p3": "ascend310p",
    "ascend310p5": "ascend310p",
    "ascend310p7": "ascend310p",
    "ascend310p3vir01": "ascend310p",
    "ascend310p3vir02": "ascend310p",
    "ascend310p3vir04": "ascend310p",
    "ascend310p3vir08": "ascend310p",
    "ascend310b": "ascend310b",
    "ascend310b1": "ascend310b",
    "ascend310b2": "ascend310b",
    "ascend310b3": "ascend310b",
    "ascend310b4": "ascend310b",
}


def get_config():
    """get config from user"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--op_dirs", type=str, required=True)
    parser.add_argument("--build_type", type=str, default="Release")
    parser.add_argument("--build_path", type=str, default="")
    parser.add_argument("--soc_version", type=str, default="")
    parser.add_argument("--ascend_cann_package_path", type=str, default="")
    parser.add_argument("--vendor_name", type=str, default="customize")
    parser.add_argument("--install_path", type=str, default="")
    parser.add_argument("-c", "--clear", action="store_true")
    parser.add_argument("-i", "--install", action="store_true")
    return parser.parse_args()


class CustomOPCompiler():
    """
    Custom Operator Offline Compilation
    """

    def __init__(self, args):
        self.args = args
        if self.args.build_path != "":
            self.custom_project = os.path.join(self.args.build_path, "CustomProject")
        else:
            self.custom_project = os.path.join(os.path.dirname(os.path.realpath(__file__)), "CustomProject")
        self.op_dirs = re.split(r"[;, ]", self.args.op_dirs)

    def check_args(self):
        """check config"""
        for op_dir in self.op_dirs:
            if not os.path.isdir(op_dir):
                raise ValueError(
                    f"Config error! op directpry [{op_dir}] is not exist, "
                    f"please check your set --op_dirs")

        if self.args.soc_version != "":
            soc_version_list = re.split(r"[;,]", self.args.soc_version)
            for soc_version in soc_version_list:
                if soc_version.lower() not in SOC_VERSION_MAP.keys():
                    raise ValueError(
                        f"Config error! Unsupported soc version(s): {soc_version}! "
                        f"Please check your set --soc_version and use ';' or ',' to separate multiple soc_versions. "
                        f"Supported soc version : {SOC_VERSION_MAP.keys()}.")

        if self.args.ascend_cann_package_path != "":
            if not os.path.isdir(self.args.ascend_cann_package_path):
                raise ValueError(
                    f"Config error! ascend cann package path [{self.args.ascend_cann_package_path}] is not valid path, "
                    f"please check your set --ascend_cann_package_path")

        if self.args.install or self.args.install_path != "":
            if self.args.install_path == "":
                opp_path = os.environ.get('ASCEND_OPP_PATH')
                if opp_path is None:
                    raise ValueError(
                        "Config error! Can not find install path, please set install path by --install_path")
                self.args.install_path = opp_path

            os.makedirs(self.args.install_path, exist_ok=True)
    
    def exec_shell_command(self, command, stdout=None):
        try:
            result = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, shell=False, text=True, check=True)
        except FileNotFoundError as e:
            logger.error(f"Command not found: {e}")
            raise RuntimeError(f"Command not found: {e}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Run {command} Command failed with return code {e.returncode}: {e.output}")
            raise RuntimeError(f"Run {command} Command failed with return code {e.returncode}: {e.output}")
        return result

    def init_config(self):
        """initialize config"""
        if self.args.ascend_cann_package_path == "":
            self.args.ascend_cann_package_path = os.environ.get('ASCEND_HOME_PATH', "/usr/local/Ascend/ascend-toolkit/latest")

        if self.args.soc_version == "":
            self.args.soc_version = "ascend910b1,ascend310p1"

    def copy_code_file(self):
        """copy code file to custom project"""
        for op_dir in self.op_dirs:
            op_host_dir = os.path.join(op_dir, OP_HOST)
            op_kernel_dir = os.path.join(op_dir, OP_KERNEL)
            if not os.path.exists(op_host_dir) or not os.path.exists(op_kernel_dir):
                logger.warning(f"The {op_dir} dose not contain {op_host_dir} or {op_kernel_dir}, skipped!")
                continue

            for item in os.listdir(op_host_dir):
                if item.split('.')[-1] in code_suffix:
                    item_path = os.path.join(op_host_dir, item)
                    target_path = os.path.join(self.custom_project, OP_HOST, item)
                    if os.path.isfile(item_path):
                        shutil.copy(item_path, target_path)

            for item in os.listdir(op_kernel_dir):
                if item.split('.')[-1] in code_suffix:
                    item_path = os.path.join(op_kernel_dir, item)
                    target_path = os.path.join(self.custom_project, OP_KERNEL, item)
                    if os.path.isfile(item_path):
                        shutil.copy(item_path, target_path)

        for root, _, files in os.walk(self.custom_project):
            for f in files:
                _, file_extension = os.path.splitext(f)
                if file_extension == ".sh":
                    os.chmod(os.path.join(root, f), 0o700)

    def trans_soc_version(self, soc_version_args):
        soc_version_list = re.split(r"[;,]", soc_version_args)
        if len(soc_version_list) == 1:
            version_map = {"ascend910": "ascend910a",
                           "ascend910b": "ascend910b1",
                           "ascend310p": "ascend310p1",
                           "ascned310b": "ascend310b1",
                           "ascend910c": "ascend910_9391"}
            soc = soc_version_list[0].lower()
            return f"ai_core-{version_map.get(soc, soc)}"

        socs = []
        for soc_version in soc_version_list:
            soc = SOC_VERSION_MAP.get(soc_version.lower())
            socs.append(soc)
        return ",".join(f"ai_core-{soc}" for soc in socs)

    def generate_compile_project(self):
        """generate compile project"""
        if os.path.exists(self.custom_project) and os.path.isdir(self.custom_project):
            shutil.rmtree(self.custom_project)

        compute_unit = self.trans_soc_version(self.args.soc_version)
        json_data = [{"op": "CustomOP"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_json = os.path.join(temp_dir, "custom.json")
            with open(custom_json, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)

            gen_command = ["msopgen", "gen", "-i", custom_json, "-c", compute_unit, "-lan", "cpp", "-out", self.custom_project]
            self.exec_shell_command(gen_command)

        if self.args.build_type.lower() == "debug":
            debug_command = ["sed", "-i", "s/Release/Debug/g", f"{self.custom_project}/CMakePresets.json"]
            self.exec_shell_command(debug_command)

        if os.getenv("CMAKE_THREAD_NUM"):
            thread_num = int(os.getenv("CMAKE_THREAD_NUM"))
            cmake_j_command = ["sed", "-i", f"s/-j$(nproc)/-j{thread_num}/g", f"{self.custom_project}/build.sh"]
            self.exec_shell_command(cmake_j_command)

        op_host_dir = os.path.join(self.custom_project, OP_HOST)
        for item in os.listdir(op_host_dir):
            if item.split('.')[-1] in code_suffix:
                os.remove(os.path.join(op_host_dir, item))

        op_kernel_dir = os.path.join(self.custom_project, OP_KERNEL)
        for item in os.listdir(op_kernel_dir):
            if item.split('.')[-1] in code_suffix:
                os.remove(os.path.join(op_kernel_dir, item))
        
        self.copy_code_file()
    
    def compile_custom_op(self):
        """compile custom operator"""
        if self.args.ascend_cann_package_path != "":
            cann_package_path = self.args.ascend_cann_package_path
            setenv_path = os.path.join(cann_package_path, "bin", "setenv.bash")
            bash_cmd = (
                f"source {setenv_path} > /dev/null 2>&1 && "
                f"export LD_LIBRARY_PATH={cann_package_path}/lib64:$LD_LIBRARY_PATH && "
                f"cd {self.custom_project} && "
                f"bash build.sh"
            )
        else:
            bash_cmd = (
                f"cd {self.custom_project} && "
                f"bash build.sh"
            )
        args = ['bash', '-c', bash_cmd]
        self.exec_shell_command(args)
        logger.info("Custom operator compiled successfully!")

    def install_custom_op(self):
        """install custom run"""
        if self.args.install or self.args.install_path != "":
            logger.info("Install custom opp run in {}".format(self.args.install_path))
            os.environ['ASCEND_CUSTOM_OPP_PATH'] = self.args.install_path
            run_path = []
            build_out_path = os.path.join(self.custom_project, "build_out")
            for item in os.listdir(build_out_path):
                if item.split('.')[-1] == "run":
                    run_path.append(os.path.join(build_out_path, item))
            if not run_path:
                raise RuntimeError("There is no custom run in {}".format(build_out_path))
            self.exec_shell_command(['bash', run_path[0]])
            logger.info("Install custom run opp successfully!")
            logger.info(
                "Please set [source ASCEND_CUSTOM_OPP_PATH={}/vendors/{}:$ASCEND_CUSTOM_OPP_PATH] to "
                "make the custom operator effective in the current path.".format(
                    self.args.install_path, self.args.vendor_name))

    def clear_compile_project(self):
        """clear log and build out"""
        if self.args.clear:
            command = ['rm', '-rf', self.custom_project]
            self.exec_shell_command(command)
            logger.info("Clear custom compile project successfully!")

    def compile(self):
        """compile op"""
        self.check_args()
        self.init_config()
        self.generate_compile_project()
        self.compile_custom_op()
        self.install_custom_op()
        self.clear_compile_project()


if __name__ == "__main__":
    config = get_config()
    custom_op_compiler = CustomOPCompiler(config)
    custom_op_compiler.compile()
