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


def get_config():
    """get config from user"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--op_host_path", type=str, required=True)
    parser.add_argument("-k", "--op_kernel_path", type=str, required=True)
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
        script_path = os.path.realpath(__file__)
        dir_path, _ = os.path.split(script_path)
        self.current_path = dir_path
        self.custom_project = os.path.join(dir_path, "CustomProject")

    def check_args(self):
        """check config"""
        if not os.path.isdir(self.args.op_host_path):
            raise ValueError(
                f"Config error! op host path [{self.args.op_host_path}] is not exist,"
                f" please check your set --op_host_path")

        if not os.path.isdir(self.args.op_kernel_path):
            raise ValueError(
                f"Config error! op kernel path [{self.args.op_kernel_path}] is not exist, "
                f"please check your set --op_kernel_path")

        if self.args.soc_version != "":
            support_soc_version = {"Ascend910", "Ascend910B", "Ascend310P", "Ascend310B"}
            input_socs = re.split(r"[;,]", self.args.soc_version)
            input_socs = [soc.strip() for soc in input_socs if soc.strip()]
            support_soc_version_lower = {soc.lower() for soc in support_soc_version}
            invalid_socs = [soc for soc in input_socs if soc.lower() not in support_soc_version_lower]
            if invalid_socs:
                raise ValueError(
                    f"Config error! Unsupported soc version(s): {invalid_socs}! "
                    f"Please check your set --soc_version and use ';' or ',' to separate multiple soc_versions, "
                    f"supported soc versions are {support_soc_version}")

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
            self.args.soc_version = "Ascend910,Ascend910B,Ascend310P,Ascend310B"

    def copy_code_file(self):
        """copy code file to custom project"""

        for item in os.listdir(self.args.op_host_path):
            if item.split('.')[-1] in code_suffix:
                item_path = os.path.join(self.args.op_host_path, item)
                target_path = os.path.join(self.custom_project, OP_HOST, item)
                if os.path.isfile(item_path):
                    shutil.copy(item_path, target_path)
        for item in os.listdir(self.args.op_kernel_path):
            if item.split('.')[-1] in code_suffix:
                item_path = os.path.join(self.args.op_kernel_path, item)
                target_path = os.path.join(self.custom_project, OP_KERNEL, item)
                if os.path.isfile(item_path):
                    shutil.copy(item_path, target_path)

        for root, _, files in os.walk(self.custom_project):
            for f in files:
                _, file_extension = os.path.splitext(f)
                if file_extension == ".sh":
                    os.chmod(os.path.join(root, f), 0o700)

    def generate_compile_project(self):
        """generate compile project"""
        if os.path.exists(self.custom_project) and os.path.isdir(self.custom_project):
            shutil.rmtree(self.custom_project)

        compute_unit = ",".join([f"ai_core-{soc}" for soc in re.split(r"[;,]", self.args.soc_version)])
        json_data = [{"op": "CustomOP"}]
        with tempfile.TemporaryDirectory() as temp_dir:
            custom_json = os.path.join(temp_dir, "custom.json")
            with open(custom_json, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)

            gen_command = ["msopgen", "gen", "-i", custom_json, "-c", compute_unit, "-lan", "cpp", "-out", self.custom_project]
            self.exec_shell_command(gen_command)
        
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
