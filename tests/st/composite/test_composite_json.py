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
import pytest
import logging
from akg import composite
from akg.utils import custom_tiling
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import gpu_profiling
from akg.utils.format_transform import to_tvm_nd_array
from tests.common.gen_json_data import gen_json_data
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor

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
    logging.info(template.format("", "explicitly enable (--mind-trick-enable=1) or disable (--mind-trick-enable=0) mind tricks"))
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


def _compare_func(output, expect):
    rtol, atol = get_rtol_atol("FUSED", str(output.dtype))
    return compare_tensor(output, expect, rtol=rtol, atol=atol)


def get_result(desc, poly, attrs=None, profiling=True):
    backend = _get_backend(desc)
    if attrs is None:
        attrs = {}

    build_attrs = attrs if attrs else None
    mod = composite.build(desc, build_attrs, poly=poly)

    input_for_mod, expect, output_indexes = gen_json_data(desc)
    output = utils.mod_launch(mod, input_for_mod, output_indexes)

    if not all(map(_compare_func, output if isinstance(output, (list, tuple)) else [output],
                   expect if isinstance(expect, (list, tuple)) else [expect])):
        logging.info(mod.imported_modules[0].get_source())
        return False
    if profiling and backend == "cuda":
        inputs = to_tvm_nd_array(input_for_mod)
        expect = to_tvm_nd_array(expect)
        gpu_profiling(mod, *inputs, *expect, repeat_time=400)
    return True


@pytest.mark.skip
def test_single_file(input_file, attrs, poly, profiling=True):
    if not input_file.endswith(".info") and not input_file.endswith(".json"):
        print("Skip {}, only process file with .info or .json suffix".format(input_file))
        return
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%file: ", input_file)
    with open(input_file, 'r') as f:
        desc = f.read()
        if get_result(desc, poly, attrs, profiling):
            logging.info("Run Pass!")
        else:
            logging.info("Precision Error")
            raise ValueError("Precision Error")


@pytest.mark.skip
def test_json_dir(poly, use_custom):
    json_dir = "./json_dir/"
    json_dims_file = "./json_dir/dims.json"
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
            logging.info("Begin run No.%d file:%s" % (idx, input_file))
            idx = idx + 1
            desc = f.read()
            attrs = dims_dict.get(input_file, {}) if use_custom else {}
            if not get_result(desc, poly, attrs):
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
        test_json_dir(poly, use_custom)
    elif ci_test:
        poly = False
        test_ci(use_profiling, poly)
    else:
        print_usage()


if __name__ == "__main__":
    main(sys.argv[1:])
