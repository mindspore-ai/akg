# Copyright 2023 Huawei Technologies Co., Ltd
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

"""AKG-MLIR Driver for MindSpore."""
import argparse
import hashlib
import json
import logging
import os
import pathlib
import subprocess
import shutil

from .utils.cpu_profiling_wrapper import wrap_timer_func

HOST_SHAPES = "hostShapes"
DEVICE_SHAPES = "deviceShapes"
RUNTIME_VARS = "runtimeVars"
SUPPORT_INFO = "SupportInfo"
TARGET_INFO = "targetInfo"
DYNAMIC = "is_dynamic"
SHA256 = "sha256"
KERNEL_NAME = "kernelName"
STATIC_TILE_IMPL = "StaticTileImpl"

def set_ascend_info(core_type, title_dict):
    if len(core_type) == 0:
        return
    if core_type == "MIX":
        title_dict["magic"] = "RT_DEV_BINARY_MAGIC_ELF"
        title_dict["coreType"] = "MIX"
        title_dict["intercoreSync"] = 1
        title_dict["taskRation"] = "1:2"
    elif core_type == "AiCore":
        title_dict["coreType"] = "AiCore"
        title_dict["magic"] = "RT_DEV_BINARY_MAGIC_ELF_AICUBE"
    elif core_type == "VectorCore":
        title_dict["coreType"] = "VectorCore"
        title_dict["magic"] = "RT_DEV_BINARY_MAGIC_ELF_AIVEC"

def write_code(js_dict, fname):
    """
    Export kernel config files.

    Args:
        js_dict: dict of kernel informations.
        fname: the name of json file to be generated.
    """
    if os.path.exists(fname):
        os.remove(fname)
    with os.fdopen(os.open(fname, os.O_WRONLY | os.O_CREAT, 0o400), "w") as f:
        json.dump(js_dict, f, sort_keys=True, indent=4, separators=(",", ":"))

def get_kernel_meta_path():
    """Return the PATH of kernel meta files."""
    kernel_meta_dir = os.getenv("KERNEL_META_DIR", default="akg_kernel_meta")
    return os.path.join(
        os.path.realpath(os.getenv("MS_COMPILER_CACHE_PATH", "")),
        kernel_meta_dir,
    )

def _is_single_op(desc_d):
    input_lists = desc_d.get("op_desc", [])
    return len(input_lists) <= 1

def generate_unique_hash(input_str):
    unique_hash = hashlib.md5(input_str.encode("utf8")).hexdigest()
    return unique_hash

def deal_input(desc):
    for input_desc in desc["input_desc"] if desc.get("input_desc") is not None else []:
        if len(input_desc[0]["shape"]) == 1 and input_desc[0]["shape"][0] == 1 and "value" in input_desc[0]:
            input_desc[0]["value"] = 0

def del_value(desc):
    for operation in desc["op_desc"]:
        deal_input(operation)
    desc["op"] = ""

def get_npucompiler_path():
    npu_compiler_path = shutil.which("bishengir-compile")
    if npu_compiler_path is None:
        raise EnvironmentError("Couldn't find executable bishengir-compile.")
    return npu_compiler_path

class AkgMlirDriver(object):
    """class AkgMlirDriver."""

    def __init__(
        self,
        input_file: str,
        output_dir: str = "",
        akg_tools_dir: str = "",
        llvm_tools_dir: str = "",
        dynamic_shape: bool = False,
        log_level: bool = "INFO",
        dump_ir=False,
        repo_path: str = "",
        profiling_trails=0,
        runtime_provider="MindSpore",
        enable_akg_loop_fusion=False,
    ):
        super().__init__()

        self.input_file = input_file
        self.output_dir = get_kernel_meta_path() if output_dir == "" else output_dir
        self.akg_tools_dir = (
            os.path.dirname(os.path.abspath(__file__))
            #os.path.join(pathlib.Path(__file__).absolute().parent, "../../build/")
            if akg_tools_dir == ""
            else akg_tools_dir
        )
        self.llvm_tools_dir = (
            os.path.join(pathlib.Path(__file__).absolute().parent, "../../third-party/llvm-project/build/")
            if llvm_tools_dir == ""
            else llvm_tools_dir
        )
        self.log_level = log_level
        self.target_info = ""
        self.dump_ir = dump_ir
        self.repo_path = repo_path
        self.profiling_trails = profiling_trails
        self.runtime_provider = runtime_provider
        self.enable_akg_loop_fusion = enable_akg_loop_fusion

        with open(input_file, "r") as f:
            kernel_info = json.loads(f.read())
            self.kernel_name = kernel_info["op"]
            self.backend = "ascend" if kernel_info["process"] == "aicore" else kernel_info["process"]
            if kernel_info.get("target_info"):
                compute_capability = kernel_info.get("target_info").get("compute_capability", "7.0")
                self.target_info = "v100" if compute_capability == "7.0" else "a100"
        self.dynamic_shape = dynamic_shape

    def compile(self):
        """
        compile interface of akg-mlir. The input is the OP description json file
        generated by MindSpore GraphKernel Module
        """
        with os.fdopen(os.open(self.input_file, os.O_RDONLY, 0o755), "r") as f_info:
            kernel_info = json.loads(f_info.read())
            if (
                self.backend == "cuda"
                and self.dynamic_shape
                and _is_single_op(kernel_info)
                and "Reduce" not in self.kernel_name
            ):
                return
        if self.backend == "cpu":
            self.run_cpu()
        elif self.backend == "cuda":
            self.run_gpu()
        elif self.backend == "ascend":
            self.run_ascend()
        else:
            raise RuntimeError("Unsupported backend: " + self.backend + "!\n")

    def run_ascend(self):
        """compile ascend kernel of akg_mlir."""
        self._run_mlir_convert()
        self._run_mlir_ascend_pipeline(self.dynamic_shape, self.kernel_name)
        self._run_ascend_generate_binary(self.kernel_name)

    def run_cpu(self):
        """compile cpu kernel of akg-mlir."""
        self._run_mlir_convert()
        self._run_mlir_cpu_pipeline(self.dynamic_shape, self.kernel_name)
        self._run_mlir_to_llvm(self.kernel_name)
        self._run_cpu_generate_binary(self.kernel_name)

    def run_gpu(self):
        """compile gpu kernel of akg-mlir."""

        def _build(is_dyn, kernel_name, tiling_mode):
            self._run_mlir_gpu_pipeline(is_dyn, kernel_name, tiling_mode)
            self._run_mlir_gpu_codegen(kernel_name)
            self._run_gpu_translate(kernel_name)
            self._run_ptx_replace(is_dyn, kernel_name)
            self._run_ptx_dump_json(is_dyn, kernel_name)

        def _gen_static_tile_kernel(kernel_name):
            try:
                sub_kernel_name = kernel_name + "_static"
                sub_input_file = os.path.join(self.output_dir, sub_kernel_name + ".info")
                sub_input_file_desc = dict()
                with open(self.input_file, "r") as f:
                    sub_input_file_desc = json.loads(f.read())
                    sub_input_file_desc["op"] = sub_kernel_name
                with os.fdopen(os.open(sub_input_file, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                    f.write(json.dumps(sub_input_file_desc))
                self._run_mlir_convert(sub_kernel_name, sub_input_file)
                logging.info("Start to build %s", sub_kernel_name)
                _build(True, sub_kernel_name, "static")
                logging.info("Success to build %s", sub_kernel_name)
            except RuntimeError as exc:
                logging.info("Fail to build %s : %s", sub_kernel_name, exc)
                if self.log_level == "ERROR":
                    raise RuntimeError(f"Compile error, kernel: {sub_kernel_name} is not generated") from exc
                logging.info("Compile error, kernel: %s", sub_kernel_name)

        default_tiling_mode = None
        if self.dynamic_shape and os.environ.get("MLIR_TILING_MODE", "auto") == "both":
            _gen_static_tile_kernel(self.kernel_name)
            default_tiling_mode = "auto"
        try:
            self._run_mlir_convert()
            _build(self.dynamic_shape, self.kernel_name, default_tiling_mode)
        except RuntimeError as exc:
            if self.log_level == "ERROR":
                raise RuntimeError(f"Compile error, kernel: {self.kernel_name} is not generated") from exc
            logging.info(f"Compile error, kernel: {self.kernel_name}")

    def _run_mlir_convert(self, kernel_name=None, input_file=None):
        if kernel_name is None:
            kernel_name = self.kernel_name
        if input_file is None:
            input_file = self.input_file
        out_file = os.path.join(self.output_dir, kernel_name + ".mlir")
        cmd = [
            os.path.join(self.akg_tools_dir, "bin/mindspore-translate"),
            "-json-to-mindspore",
            input_file,
            "-o",
            out_file,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("mlir pipeline failed in converting the case: " + kernel_name + "!\n")

    def _run_mlir_cpu_pipeline(self, dyn_shape, kernel_name):
        input_file = os.path.join(self.output_dir, kernel_name + ".mlir")
        out_file = os.path.join(self.output_dir, kernel_name + "_out.mlir")
        cpu_opt_option = "--cpu-opt"
        if dyn_shape:
            cpu_opt_option += "=dynamic-shape=true"
        else:
            cpu_opt_option += "=cpu-outlining=false outlining-platform=" + self.runtime_provider
        cmd = [os.path.join(self.akg_tools_dir, "bin/akg-opt"), input_file, cpu_opt_option, "-o", out_file]
        if self.dump_ir:
            cmd.append("--mlir-print-ir-after-all")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if self.dump_ir:
                dump_log = os.path.join(self.output_dir, kernel_name + "_dump_cpu.log")
                with os.fdopen(os.open(dump_log, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                    f.write(result.stderr)
        except subprocess.CalledProcessError:
            raise RuntimeError("mlir pipeline failed in case: " + kernel_name + "!\n")

        logging.info("mlir pipeline success")
        return

    def _run_mlir_ascend_pipeline(self, dyn_shape, kernel_name):
        input_file = os.path.join(self.output_dir, kernel_name + ".mlir")
        out_file = os.path.join(self.output_dir, kernel_name + "_out.mlir")
        ascend_opt_option = "--ascend-opt"
        if dyn_shape:
            ascend_opt_option += "=dynamic-shape=true"
        if self.enable_akg_loop_fusion:
            ascend_opt_option += "=enable-akg-loop-fusion=1"
        cmd = [os.path.join(self.akg_tools_dir, "bin/akg-opt"), input_file, ascend_opt_option, "-o", out_file]
        if self.dump_ir:
            cmd.append("--mlir-print-ir-after-all")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if self.dump_ir:
                dump_log = os.path.join(self.output_dir, kernel_name + "_dump_ascend_state1.log")
                with os.fdopen(os.open(dump_log, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                    f.write(result.stderr)
        except subprocess.CalledProcessError:
            raise RuntimeError("mlir pipeline failed in case: " + kernel_name + "!\n")

        logging.info("mlir pipeline success")
        return

    def _dump_ascend_meta_data(self, block_dim, kernel_name):
        logging.info("dump ascend meta data:")
        title_dict = dict()
        # ascend info
        set_ascend_info("VectorCore", title_dict)
        title_dict["kernelName"] = kernel_name
        # thread info
        title_dict["blockDim"] = block_dim
        # bin file info
        bin_file_suffix = ".so"
        title_dict["binFileSuffix"] = bin_file_suffix
        bin_file_name = "lib" + kernel_name
        title_dict["binFileName"] = bin_file_name
        # sha256
        buf_size = 64 * 1024  # once read 64kb
        root_path = get_kernel_meta_path()
        sha256 = hashlib.sha256()
        kernel_file_name = os.path.join(self.output_dir, bin_file_name + bin_file_suffix)
        with open(kernel_file_name, "rb") as kf:
            while True:
                data = kf.read(buf_size)
                if not data:
                    break
                sha256.update(data)
        title_dict["sha256"] = sha256.hexdigest()  # sha256

        json_file = os.path.join(self.output_dir, kernel_name + ".json")
        write_code(title_dict, json_file)
        return

    def _run_ascend_generate_binary(self, kernel_name):
        logging.info("bishengir-compile code generater:")
        npu_compiler_path = get_npucompiler_path()
        input_file = os.path.join(self.output_dir, kernel_name + "_out.mlir")
        out_file = os.path.join(self.output_dir, kernel_name + ".so")

        cmd = [
            npu_compiler_path,
            input_file,
            "-enable-hfusion-compile=true",
            "-enable-hivm-compile=true",
            "-enable-bin-relocation=false",
            "-block-dim=40",
            "-enable-auto-multi-buffer=true",
            "-o",
            out_file,
        ]

        if self.dump_ir:
            cmd.append("--mlir-print-ir-after-all")
        dump_log = os.path.join(self.output_dir, kernel_name + "_dump_bishengir.log")
        with os.fdopen(os.open(dump_log, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
            try:
                result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                f.write(result.stderr)
            except subprocess.CalledProcessError as e:
                f.write(str(e))
                raise RuntimeError("generate ascend binary: " + input_file + "!\n")
        logging.info("generate ascend binary success")

        self._dump_ascend_meta_data(block_dim=20, kernel_name=self.kernel_name)
        return

    def _run_mlir_to_llvm(self, kernel_name):
        logging.info("mlir to llvm:")
        input_file = os.path.join(self.output_dir, kernel_name + "_out.mlir")
        if self.profiling_trails > 0:
            input_file = wrap_timer_func(input_file, self.kernel_name, self.profiling_trails)
        out_file = os.path.join(self.output_dir, kernel_name + ".ll")
        cmd = ["mlir-translate", input_file, "--mlir-to-llvmir", "-o", out_file]
        print("_run_mlir_to_llvm:", cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("mlir to llvm failed in case: " + input_file + "!\n")
        logging.info("mlir to llvm success")
        return

    def _run_cpu_generate_binary(self, kernel_name):
        input_file = os.path.join(self.output_dir, kernel_name + ".ll")
        out_file = os.path.join(self.output_dir, kernel_name + ".s")
        bin_file = os.path.join(self.output_dir, kernel_name + ".so")
        cmd = [
            "llc",
            input_file,
            "-relocation-model=pic",
            "-O3",
            "-o",
            out_file,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("generate .s failed in case " + input_file + "!\n")

        cmd = [
            "clang++",
            out_file,
            "--rtlib=compiler-rt",
            "-fopenmp",
            "-O3",
            "--shared",
            "-fPIC",
            "-o",
            bin_file,
            "-L",
            os.path.join(self.llvm_tools_dir, "lib/"),
            "-lmlir_c_runner_utils",
        ]
        if self.runtime_provider == "MLIR":
            cmd.extend(["-L", os.path.join(self.akg_tools_dir, "lib/"), "-lmlir_akgParallelLaunch_runtime"])
        if self.profiling_trails > 0:
            cmd.extend(["-L", os.path.join(self.llvm_tools_dir, "lib/"), "-lmlir_runner_utils"])
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("generate .so failed in case " + input_file + "!\n")

        logging.info("generate cpu binary .so success")
        title_dict = dict()
        # kernel name
        title_dict[KERNEL_NAME] = kernel_name + "_kernel"
        # thread number
        thread_num = "null"
        title_dict["threadNumber"] = thread_num
        lib_file = os.path.join(self.output_dir, kernel_name + ".so")
        # sha256 of files
        lib_sha256 = hashlib.sha256()
        with open(lib_file, "rb") as f:
            lib_sha256.update(f.read())
        lib_hash_str = lib_sha256.hexdigest()
        title_dict[SHA256] = lib_hash_str

        json_file = os.path.join(self.output_dir, kernel_name + ".json")
        write_code(title_dict, json_file)

    def _run_ascend_generate_binary_(self, kernel_name):
        input_file = os.path.join(self.output_dir, kernel_name + ".ll")
        out_file = os.path.join(self.output_dir, kernel_name + ".s")
        bin_file = os.path.join(self.output_dir, kernel_name + ".so")
        cmd = [
            "llc",
            input_file,
            "-relocation-model=pic",
            "-O3",
            "-o",
            out_file,
        ]
        print("_run_ascend_generate_binary:0 ", cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("generate .s failed in case " + input_file + "!\n")

        cmd = [
            "clang++",
            out_file,
            "--rtlib=compiler-rt",
            "-O3",
            "--shared",
            "-fPIC",
            "-o",
            bin_file,
            "-L",
            os.path.join(self.llvm_tools_dir, "lib/"),
            "-lmlir_c_runner_utils",
        ]
        if self.runtime_provider == "MLIR":
            cmd.extend(["-L", os.path.join(self.akg_tools_dir, "lib/"), "-lmlir_akgParallelLaunch_runtime"])
        if self.profiling_trails > 0:
            cmd.extend(["-L", os.path.join(self.llvm_tools_dir, "lib/"), "-lmlir_runner_utils"])
        print("_run_ascend_generate_binary:1 ", cmd)
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("generate .so failed in case " + input_file + "!\n")

        logging.info("generate ascend binary .so success")
        title_dict = dict()
        # kernel name
        title_dict[KERNEL_NAME] = kernel_name + "_kernel"
        # thread number
        thread_num = "null"
        title_dict["threadNumber"] = thread_num
        lib_file = os.path.join(self.output_dir, kernel_name + ".so")
        # sha256 of files
        lib_sha256 = hashlib.sha256()
        with open(lib_file, "rb") as f:
            lib_sha256.update(f.read())
        lib_hash_str = lib_sha256.hexdigest()
        title_dict[SHA256] = lib_hash_str

        json_file = os.path.join(self.output_dir, kernel_name + ".json")
        write_code(title_dict, json_file)

    def has_reduce(self):
        """Return if the fused op contain reduction operator."""
        with open(self.input_file, "r") as f:
            desc_d = json.loads(f.read())
            for op in desc_d.get("op_desc"):
                op_name = op.get("name")
                if "reduce" in op_name.lower():
                    return True
        return False

    def _run_mlir_gpu_pipeline(self, dyn_shape, kernel_name, tiling_mode=None):
        input_file = os.path.join(self.output_dir, kernel_name + ".mlir")
        out_file = os.path.join(self.output_dir, kernel_name + "_gpu.mlir")
        opt_pipeline = "--gpu-dyn-opt" if dyn_shape else "--gpu-opt"
        opt_options = ""
        if dyn_shape:
            if tiling_mode is None:
                tiling_mode = os.environ.get("MLIR_TILING_MODE", "auto")
            opt_options += "tiling-mode=" + tiling_mode

        if os.path.exists(self.repo_path):
            opt_options += " global-config-file=" + self.repo_path

        if opt_options != "":
            opt_pipeline += "=" + opt_options

        cmd = [self.akg_tools_dir + "bin/akg-opt", input_file, opt_pipeline, "-o", out_file]
        if self.dump_ir:
            cmd.append("--mlir-print-ir-after-all")
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if self.dump_ir:
                dump_log = os.path.join(self.output_dir, kernel_name + "_dump_gpu.log")
                with os.fdopen(os.open(dump_log, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
                    f.write(result.stderr)
        except subprocess.CalledProcessError:
            raise RuntimeError("mlir gpu pipeline failed in case: " + kernel_name + "!\n")
        logging.info("mlir gpu pipeline success: %s", kernel_name)
        return

    def _run_mlir_gpu_codegen(self, kernel_name):
        logging.info("gpu_codegen:")
        input_file = os.path.join(self.output_dir, kernel_name + "_gpu.mlir")
        out_file = os.path.join(self.output_dir, kernel_name + "_nvvm.mlir")
        cmd = [os.path.join(self.akg_tools_dir, "bin/akg-opt"), input_file, "--gpu-codegen", "-o", out_file]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("gpu_codegen failed in case: %s" + kernel_name + "!\n")
        logging.info("gpu_codegen success: %s", kernel_name)
        return

    def _run_gpu_translate(self, kernel_name):
        logging.info("mlir to ptx:")
        input_file = os.path.join(self.output_dir, kernel_name + "_nvvm.mlir")
        out_prefix = os.path.join(self.output_dir, kernel_name + "_init")
        cmd = [
            os.path.join(self.akg_tools_dir, "bin/akg-translate"),
            "-gen-ptx",
            "-arch=sm_70",
            input_file,
            "--kernel-name=" + out_prefix,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("mlir to ptx failed in case: " + kernel_name + "!\n")
        logging.info("mlir to ptx success: %s", kernel_name)

    def _run_ptx_replace(self, dyn_shape, kernel_name):
        logging.info("ptx replacement")
        input_file = os.path.join(self.output_dir, kernel_name + "_init.ptx")
        out_prefix = os.path.join(self.output_dir, kernel_name)
        out_file = out_prefix + ".ptx"
        shape_arg_file = os.path.join(self.output_dir, kernel_name + "_shape_arg.txt")
        cmd = [os.path.join(self.akg_tools_dir, "bin/akg-ptx-replace"), input_file, shape_arg_file, out_file]
        if dyn_shape:
            cmd += ["none", "dynamic_shape"]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            raise RuntimeError("ptx replacement failed in case: " + input_file + "!\n")
        logging.info("ptx replacement success: %s", kernel_name)

    def _run_ptx_dump_json(self, dyn_shape, kernel_name):
        title_dict = dict()
        json_file = os.path.join(self.output_dir, kernel_name + ".json")
        with open(json_file, "rb") as f:
            params = json.load(f)
            # Skip useless integar lists like ["32", 1] in "Seq"
            for k, v in params.items():
                if not (isinstance(v, list) and len(v) == 2 and isinstance(v[0], str) and v[0].isdigit()):
                    title_dict[k] = v
                elif "Seq" not in k:
                    title_dict[k] = (int(v[0]) - 1) // v[1] + 1

        out_file = os.path.join(self.output_dir, kernel_name + ".ptx")

        if dyn_shape:
            shape_info_json = os.path.join(self.output_dir, kernel_name + "_shape_info.json")
            if not os.path.exists(shape_info_json):
                raise RuntimeError(
                    "Dynamic shape needs file {} to get the device shape. Otherwise, the result may be \
                                    incorrect.".format(
                        shape_info_json
                    )
                )

            with os.fdopen(os.open(shape_info_json, os.O_RDONLY, 0o755), "rb") as f:
                shape_params = json.load(f)
                title_dict[HOST_SHAPES] = shape_params.get(HOST_SHAPES, [])
                title_dict[DEVICE_SHAPES] = shape_params.get(DEVICE_SHAPES, [])
                title_dict[RUNTIME_VARS] = shape_params.get(RUNTIME_VARS, [])
                title_dict[SUPPORT_INFO] = shape_params.get(SUPPORT_INFO, [])
            title_dict[TARGET_INFO] = self.target_info
            if len(title_dict[RUNTIME_VARS]) > 0:
                self._dump_static_tile_kernel_impl(kernel_name, title_dict, out_file)

        title_dict[KERNEL_NAME] = kernel_name + "_kernel"
        title_dict[DYNAMIC] = dyn_shape
        # sha256 of files
        lib_sha256 = hashlib.sha256()
        with os.fdopen(os.open(out_file, os.O_RDONLY, 0o755), "rb") as f:
            lib_sha256.update(f.read())
        title_dict[SHA256] = lib_sha256.hexdigest()

        write_code(title_dict, json_file)
        return

    def _dump_static_tile_kernel_impl(self, kernel_name, title_dict, out_file):
        static_json_file = os.path.join(self.output_dir, kernel_name + "_static.json")
        static_ptx_file = os.path.join(self.output_dir, kernel_name + "_static.ptx")
        if not (os.path.exists(static_json_file) and os.path.exists(static_ptx_file)):
            return

        with os.fdopen(os.open(static_json_file, os.O_RDONLY, 0o755), "r") as sf:
            title_dict[STATIC_TILE_IMPL] = json.load(sf)
        static_kernel_str = []
        with os.fdopen(os.open(static_ptx_file, os.O_RDONLY, 0o755), "r") as sf:
            start = False
            for line in sf:
                if ".entry" in line:
                    start = True
                if start:
                    static_kernel_str.append(line)
        with os.fdopen(os.open(out_file, os.O_WRONLY | os.O_CREAT, 0o755), "w") as f:
            for line in static_kernel_str:
                f.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run akg-mlir End to End")
    parser.add_argument("-f", type=str, help="Run single file.")
    parser.add_argument("-o", type=str, help="output dir.", default="")
    parser.add_argument("-akg-tools-dir", type=str, help="akg-mlir tools build dir.", default="")
    parser.add_argument("-llvm-tools-dir", type=str, help="llvm tools build dir", default="")
    parser.add_argument("-bisheng-tools-dir", type=str, help="bisheng cpp tools build dir", default="")
    parser.add_argument("-d", "--dynamic-shape", type=bool, help="Specifies dynamic shape or not", default=False)
    args = parser.parse_args()
    logging.info(args)

    driver = AkgMlirDriver(
        input_file=args.f,
        output_dir=args.o,
        akg_tools_dir=args.akg_tools_dir,
        llvm_tools_dir=args.llvm_tools_dir,
        dynamic_shape=args.dynamic_shape,
    )
    driver.compile()
