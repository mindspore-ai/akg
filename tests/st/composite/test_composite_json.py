# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

import os
import sys
import json
import time
import pytest
import logging
from akg import composite
from akg.utils import custom_tiling
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_json_data import gen_json_data
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor, dump_tensor

logging.getLogger().setLevel(logging.INFO)


def print_usage():
    template = "  {0:15}{1}"
    logging.info(
        "\nUsage: python test_composite_json.py [arguments] <-f/-d/--ci> [file name].\n")
    logging.info(template.format("-f", "Run single file."))
    logging.info(template.format("-d", "Run all files in json_dir."))
    logging.info(template.format("--ci", "Run files for CI testing."))
    logging.info(
        "\nNote: if no other arguments are specified, it will run according to the sets in files, or by default.")
    logging.info("\n")
    logging.info("Optional arguments:")
    logging.info(template.format("-h, --help", "Show the usage."))
    logging.info(template.format("-a, --auto", "Use poly schedule, which is enable by default."))
    logging.info(template.format("-m, --manual", "Use tvm schedule."))
    logging.info(
        template.format("-c", "Use custom attributes of tiling (defined in json_dir/dims.json) when use '-d' command."))
    logging.info(template.format("--profile", "Enable profiling when use '--ci' command."))
    logging.info(template.format("--enable_atomic_add=true or false", ""))
    logging.info(template.format("", "Set true or false to enable atomic add when use poly schedule."))
    logging.info(template.format("--dim=<args>", ""))
    logging.info(template.format("", "Set attribute of 'dim' when use '-f' command."))
    logging.info(template.format("--bind_block=<args>", ""))
    logging.info(template.format("", "Set attribute of 'bind_block' when use '-f' command."))
    logging.info(template.format("--bind_thread=<args>", ""))
    logging.info(template.format("", "Set attribute of 'bind_thread' when use '-f' command."))
    logging.info(template.format("--mind-trick-enable=<0|1>", ""))
    logging.info(
        template.format("", "explicitly enable (--mind-trick-enable=1) or disable (--mind-trick-enable=0) mind tricks"))
    logging.info(template.format("--mind-trick-file", ""))
    logging.info(template.format("", "json mind trick file"))
    logging.info(template.format("--mind-trick-string", ""))
    logging.info(template.format("", "json mind-trick string"))
    logging.info("\n")


def _get_json_dict(desc):
    return json.loads(desc) if isinstance(desc, str) else desc


def _get_backend(desc):
    json_obj = _get_json_dict(desc)
    if "process" not in json_obj.keys():
        logging.info("Can't identify the backend.")
        return None
    return json_obj["process"]


def _compare_func(output, expect, compare_tolerance=None):
    if compare_tolerance is not None:
        rtol = atol = compare_tolerance
    else:
        rtol, atol = get_rtol_atol("FUSED", str(output.dtype))
    return compare_tensor(output, expect, rtol=rtol, atol=atol)


def _dump_data(path, input, output, expect):
    input = input if isinstance(input, (list, tuple)) else [input]
    output = output if isinstance(output, (list, tuple)) else [output]
    expect = expect if isinstance(expect, (list, tuple)) else [expect]

    file_name_prefix = path + "/data/"
    if not os.path.isdir(file_name_prefix):
        os.makedirs(file_name_prefix)

    for i, data in enumerate(input):
        dump_tensor(data, file_name_prefix + 'input_' + str(i))

    for i, data in enumerate(output):
        dump_tensor(data, file_name_prefix + 'output_' + str(i))

    for i, data in enumerate(expect):
        dump_tensor(data, file_name_prefix + 'expect_' + str(i))


def _dump_info(desc, build_attrs, poly, input, output, expect):
    dump_path = os.getenv("AKG_DUMP_TESTCASE_INFO_PATH")
    if not dump_path or dump_path == "":
        # dump data to the current dir by default
        dump_path = "./DUMP"

    json_info = _get_json_dict(desc)
    op_name = json_info['op']
    today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    dump_path = dump_path + "/" + today + '/' + op_name + "/"

    dump_path = os.path.realpath(dump_path)
    if not os.path.isdir(dump_path):
        os.makedirs(dump_path)

    _dump_data(dump_path, input, output, expect)


class IOInfo(object):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    def __str__(self):
        return "(" + str(self.name) + ", " + str(self.dtype) + ")"

    def __repr__(self):
        return "(" + str(self.name) + ", " + str(self.dtype) + ")"

    def __hash__(self):
        return hash(self.name + self.dtype)

    def __eq__(self, other):
        if not isinstance(other, IOInfo):
            return False
        return self.name == other.name and self.dtype == other.dtype


def precision_analyze(desc: dict, tensors):
    exclude_op_list = ["Minimum", "Maximum", "Reshape", "ZerosLike", "Tile", "Select", "InplaceAssign", "Greater",
                       "SelectGT", "SelectLT", "LessEqual", "Less", "EquivFormat", "ExpandDims", "Transpose",
                       "TransData", "BroadcastTo", "Assign"]
    input_tensors = []
    for input_desc in desc["input_desc"] if desc["input_desc"] is not None else []:
        input_tensors.append(IOInfo(input_desc[0]["tensor_name"], input_desc[0]["data_type"]))

    # Construct graph of current json
    graph = {}
    ops = {}  # recorder the operator that generates the current output
    for op in desc["op_desc"]:
        if op["name"] == "InplaceAssign":
            output = IOInfo(op["input_desc"][0][0]["tensor_name"], op["input_desc"][0][0]["data_type"])
            input = IOInfo(op["input_desc"][1][0]["tensor_name"], op["input_desc"][1][0]["data_type"])
            graph[output] = [input]
            ops[output] = op["name"]
            fake_output = False
            for attr in op["attr"]:
                if attr["name"] == "fake_output":
                    fake_output = attr["value"]
            if not fake_output:
                output = IOInfo(op["output_desc"][0]["tensor_name"], op["output_desc"][0]["data_type"])
                input = IOInfo(op["input_desc"][2][0]["tensor_name"], op["input_desc"][2][0]["data_type"])
                graph[output] = [input]
                ops[output] = op["name"]
        else:
            output = IOInfo(op["output_desc"][0]["tensor_name"], op["output_desc"][0]["data_type"])
            inputs = []
            for input_desc in op["input_desc"]:
                inputs.append(IOInfo(input_desc[0]["tensor_name"], input_desc[0]["data_type"]))
            graph[output] = inputs
            ops[output] = op["name"]

    def _precision_reduce(x: IOInfo):
        if x in input_tensors:
            logging.debug("Skip {}, because it comes from input tensors".format(x))
            return False
        if x in ops and ops[x] in exclude_op_list:
            logging.debug("Skip {}, because it comes from [{}] that will not reduce precision".format(x, ops[x]))
            return False
        return True

    # DFS search for each tensor in tensors to check if they are casted from fp16
    tensors = tensors if isinstance(tensors, (list, tuple)) else [tensors]
    cast_from_fp16 = [False for _ in range(len(tensors))]
    for i, tensor in enumerate(tensors):
        logging.debug("[{}/{}] Analyzing sources of {}...".format(i + 1, len(tensors), tensor))
        visited = []
        stack = [tensor]
        while len(stack) > 0:
            cur_tensor = stack[-1]
            stack.pop()
            # If output comes from a fp16 tensor, and this tensor is produced by operators that will increase
            # the rounding errors(such as Add), then we mark the output cast_from_fp16
            if cur_tensor.dtype == "float16" and _precision_reduce(cur_tensor):
                logging.info("{} --> {}".format(tensor, cur_tensor))
                cast_from_fp16[i] = True
                break
            if cur_tensor not in visited:
                visited.append(cur_tensor)
                if cur_tensor not in graph:
                    continue
                for t in graph[cur_tensor]:
                    stack.append(t)
    return cast_from_fp16


def get_compare_tolerance(json_str: str, output_indexes: list):
    desc = json.loads(json_str)
    compare_tolerance = []

    # Collects output tensors of current json based on output_indexes.
    io = []
    for input_desc in desc["input_desc"] if desc["input_desc"] is not None else []:
        io.append(IOInfo(input_desc[0]["tensor_name"], input_desc[0]["data_type"]))
    for output_desc in desc["output_desc"]:
        io.append(IOInfo(output_desc["tensor_name"], output_desc["data_type"]))
    outputs = [io[idx] for idx in output_indexes]

    analyze_indexes = []  # holds the index of float32 outputs
    for i, o in enumerate(outputs):
        if o.dtype == "float16":
            compare_tolerance.append(1e-3)
        else:
            compare_tolerance.append(1e-4)
            if o.dtype == "float32":
                analyze_indexes.append(i)
    if not analyze_indexes:
        return compare_tolerance

    # Check if these float32 outputs are cast from fp16, if so, use 1e-3 rtol and atol when comparing with numpy
    logging.debug("=============== Precision Analyze ===============")
    analyze_outputs = [outputs[idx] for idx in analyze_indexes]
    cast_from_fp16 = precision_analyze(desc, analyze_outputs)
    for i, v in enumerate(cast_from_fp16):
        if v:
            compare_tolerance[analyze_indexes[i]] = 1e-3
            logging.warning("{} will use tolerance {} when comparing with expect."
                            .format(outputs[analyze_indexes[i]], compare_tolerance[analyze_indexes[i]]))
    logging.debug("============= Precision Analyze End =============")

    return compare_tolerance


def get_result(desc, poly, attrs=None, profiling=True, need_compare=True):
    backend = _get_backend(desc)

    mod = composite.build(desc, attrs, poly=poly)
    if not need_compare:
        return True
    input_for_mod, expect, output_indexes = gen_json_data(desc)
    output = utils.mod_launch(mod, input_for_mod, output_indexes)

    compare_tolerance = get_compare_tolerance(desc, output_indexes)
    compare_res = list(map(_compare_func, output if isinstance(output, (list, tuple)) else [output],
                           expect if isinstance(expect, (list, tuple)) else [expect], compare_tolerance))
    if not all(compare_res):
        logging.debug(mod.imported_modules[0].get_source())
        _dump_info(desc, attrs, poly, input_for_mod, output, expect)
        logging.warning("Compare results: {}".format(compare_res))
        return False
    if profiling and backend == "cuda":
        inputs = to_tvm_nd_array(input_for_mod)
        expect = to_tvm_nd_array(expect)
        gpu_profiling(mod, *inputs, *expect, repeat_time=400)
    return True


@pytest.mark.skip
def test_single_file(input_file, attrs, poly, profiling=True, max_run_times=3):
    if not input_file.endswith(".info") and not input_file.endswith(".json"):
        print("Skip {}, only process file with .info or .json suffix".format(input_file))
        return
    logging.info("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%file: {}".format(input_file))
    with open(input_file, 'r') as f:
        desc = f.read()
        for i in range(max_run_times):
            if get_result(desc, poly, attrs, profiling):
                logging.info("Run Pass! max run time: {}, current run time: {}".format(max_run_times, i + 1))
                return
            logging.info("Precision error! max run time: {}, current run time: {}".format(max_run_times, i + 1))
        raise ValueError("Precision Error")


@pytest.mark.skip
def test_json_dir(poly, use_custom, json_dir="./json_dir/", online_tuning=0):
    json_dims_file = json_dir + "dims.json"
    dims_dict = {}
    if use_custom:
        with open(json_dims_file, 'r') as f:
            base = f.read()
            dims_dict = json.loads(base)
    idx = 1
    files = os.listdir(json_dir)
    for input_file in files:
        if input_file == "dims.json":
            continue
        with open(json_dir + input_file, 'r') as f:
            logging.info("Begin run [No.%d/%d] file:%s" % (idx, len(files), input_file))
            idx = idx + 1
            desc = f.read()
            attrs = dims_dict.get(input_file, {}) if use_custom else {}
            need_compare = True
            if online_tuning:
                attrs["online_tuning"] = online_tuning
                need_compare = False
            if not get_result(desc, poly, attrs, need_compare=need_compare):
                logging.info("----------Error Json name is----------")
                logging.info(input_file)
                raise ValueError("Precision Error")
    logging.info("All Json files run PASS!")


def get_op_cycles_info(desc, cycle_info_file, old_op_cycles=100000000):
    with open(cycle_info_file, 'r') as f:
        op_cycles = int(f.read())
    diff = old_op_cycles - op_cycles
    return op_cycles, diff


def test_ci(profile=False, poly=False):
    pwd = os.path.dirname(os.path.abspath(__file__))
    ci_path = pwd + "/need_adapt/"
    target_process = ["cuda", "aicore"]
    if profile:
        need_update = False
        base_json_file = pwd + "/need_adapt/base.json"
        cycle_info_file = pwd + "/cycle_path/a.txt"
        os.environ['PROFILING'] = "true"
        os.environ['CYCLES_PATH'] = os.getcwd() + '/' + cycle_info_file
        with open(base_json_file, 'r') as f:
            base = f.read()
            old_dict = json.loads(base)
    files = os.listdir(ci_path)
    for fi in files:
        with open(ci_path + fi, 'r') as f:
            if fi == "base.json":
                continue
            desc = f.read()
            json_desc = json.loads(desc)
            if "process" not in json_desc or json_desc["process"] not in target_process:
                logging.info("------ Skip {}".format(fi))
                continue
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  fi: ", fi)
            flag = get_result(desc, poly)
            if not flag:
                logging.info("----------Error Json info is----------")
                logging.info(desc)
                raise ValueError("Precision Error")
            elif not profile:
                logging.info("Composite Json {} pass!".format(fi))
            else:
                old_op_cycles = old_dict[fi]
                op_cycles, diff = get_op_cycles_info(
                    desc, cycle_info_file, old_op_cycles)
                logging.info("~~~~~~~~~~~cycle diff is~~~~~~~~~~~")
                logging.info(diff)
                if diff > 500:
                    need_update = True
                    logging.info("Find Better Cycle the Json Info is:")
                    logging.info(desc)
                    logging.info("The Better Cycle is:")
                    logging.info(op_cycles)
                    old_dict[fi] = op_cycles
                elif diff < -1000:
                    logging.info("----------Error Json info is----------")
                    logging.info(desc)
                    raise ValueError("Performance Degradation")
            assert (flag)
    logging.info("All ops are ok!")
    if profile:
        if need_update:
            logging.info("Need to Update Baseline!!!")
            with open(base_json_file, 'w', encoding='utf-8') as f:
                json.dump(old_dict, f, indent=4)
        else:
            logging.info(
                "No significant performance improvement. Do not need to update Baseline!")


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ci_ascend():
    test_ci()


def main(argv):
    import getopt
    try:
        options, args = getopt.getopt(argv, "atdcf:mh", ["auto", "manual", "ci", "profile", "tune"
                                                         "enable_atomic_add=", "dim=", "bind_block=", "bind_thread=",
                                                         "mind-trick-enable=", "mind-trick-file=", "mind-trick-string=",
                                                         "help"])
        poly = True
        single_file = False
        dir_test = False
        use_custom = False
        ci_test = False
        use_profiling = False
        online_tuning = 0
        attrs_list = {}
        for option, value in options:
            if option in ("-h", "--help"):
                print_usage()
                sys.exit()
            if option in ("-a", "--auto"):
                poly = True
            elif option in ("-m", "--manual"):
                poly = False
            elif option in ("-t", "--tune"):
                online_tuning = 1
                attrs_list["online_tuning"] = 1
            elif option == "--ci":
                ci_test = True
            elif option == "--profile":
                use_profiling = True
            elif option == "-d":
                dir_test = True
            elif option == "-c":
                use_custom = True
            elif option == "-f":
                single_file = True
                file_name = value
            elif option == "--enable_atomic_add":
                attrs_list["enable_atomic_add"] = True if value == "true" else False
            elif option == "--dim":
                attrs_list["dim"] = value
            elif option == "--bind_block":
                attrs_list["bind_block"] = value
            elif option == "--bind_thread":
                attrs_list["bind_thread"] = value
            elif option == "--mind-trick-enable":
                attrs_list['enable_mind_trick'] = int(value)
            elif option == "--mind-trick-file":
                with open(value, 'r') as f:
                    attrs_list['mind_trick'] = f.read()
            elif option == "--mind-trick-string":
                attrs_list['mind_trick'] = value
    except:
        print_usage()
        return

    if single_file:
        test_single_file(file_name, attrs_list, poly)
    elif dir_test:
        if len(args) == 1:
            all_json_path = list()
            json_path = args[0]
            if os.path.exists(json_path) and os.path.isdir(json_path):
                files = os.listdir(json_path)
                if len(files) > 0 and os.path.isfile(json_path + files[0]):
                    all_json_path.append(json_path)
                else:
                    # parent dir
                    for f in files:
                        all_json_path.append(json_path + f + "/")
            for p in all_json_path:
                test_json_dir(poly, use_custom, p, online_tuning)
        else:
            test_json_dir(poly, use_custom, online_tuning)

    elif ci_test:
        poly = False
        test_ci(use_profiling, poly)
    else:
        print_usage()


if __name__ == "__main__":
    main(sys.argv[1:])
