# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
from copy import copy
import os
import sys
import json
import time
import functools
import glob
import logging
from multiprocessing import Process
import pytest
import numpy as np
import akg
import akg.tvm as tvm
from akg import composite
from akg.utils import kernel_exec as utils
from akg.utils.result_analysis import target_profiling, get_compare_tolerance
from akg.utils.format_transform import to_tvm_nd_array
from akg.utils.composite_op_helper import gen_json_data
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor, dump_tensor
from akg.ms import compilewithjson
from akg.utils.composite_op_helper import random_data_to_disk
from akg.utils.tbe_codegen_utils import copy_to_akg_kernel_meta
from akg.utils.util import get_ascend_type
logging.getLogger().setLevel(logging.INFO)


def enable_input_cache():
    if os.environ.get("RANDOM_DATA_DISK_PATH", None) is None:
        os.environ["RANDOM_DATA_DISK_PATH"] = "."
    random_files = os.environ.get("RANDOM_DATA_DISK_PATH") + "/random_data*bin"
    if len(glob.glob(random_files)) == 0:
        random_data_to_disk(size=10485760, miu=[1, 0.5, 0.1], sigma=[0.1, 0.05, 0.01])

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


def _dump_data(path, inputs, output, expect):
    inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
    output = output if isinstance(output, (list, tuple)) else [output]
    expect = expect if isinstance(expect, (list, tuple)) else [expect]

    file_name_prefix = path + "/data/"
    if not os.path.isdir(file_name_prefix):
        # pylint: disable=unexpected-keyword-arg
        os.makedirs(file_name_prefix, exist_ok=True)

    for i, data in enumerate(inputs):
        dump_tensor(data, file_name_prefix + 'input_' + str(i))

    for i, data in enumerate(output):
        dump_tensor(data, file_name_prefix + 'output_' + str(i))

    for i, data in enumerate(expect):
        dump_tensor(data, file_name_prefix + 'expect_' + str(i))


def _dump_info(desc, build_attrs, poly, inputs, output, expect):
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
        # pylint: disable=unexpected-keyword-arg
        os.makedirs(dump_path, exist_ok=True)

    logging.debug("build attrs : %s", str(build_attrs))
    logging.debug("use poly : %s", str(poly))
    _dump_data(dump_path, inputs, output, expect)



def save_profiling_csv(kernel_name, akg_cycle=-1, tbe_cycle=-1):
    import csv
    path = "./profiling.csv"
    if os.path.exists(path) is False:
        with open(path, 'w') as f:
            csv_write = csv.writer(f)
            data_row = ["kernel_name", "akg_cycle", "tbe_cycle"]
            csv_write.writerow(data_row)
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        data_row = [kernel_name, akg_cycle, tbe_cycle]
        csv_write.writerow(data_row)


def get_profiling_compare_result(desc, poly, attrs=None, profiling=True, need_compare=True):
    akg_cycles = True
    tbe_cycles = True
    result, akg_cycles = get_result(desc, poly, attrs, profiling, need_compare)
    if akg_cycles is None:
        akg_cycles_num = -1
    else:
        akg_cycles_num = akg_cycles['run_time']

    npu_pro_com = os.getenv("USE_NPU_PROFILING_COMPARE")
    if npu_pro_com is None:
        return result

    logging.info("============================= TBE =============================")

    _, tbe_cycles = get_result(desc, poly, attrs, profiling, need_compare, is_only_npu = True)
    if tbe_cycles is None:
        tbe_cycles_num = -1
    else:
        tbe_cycles_num = tbe_cycles['run_time']

    kerel_name = json.loads(desc).get("op", "test")
    save_profiling_csv(kerel_name, akg_cycles_num, tbe_cycles_num)
    return result


def get_model(desc, poly, backend, attrs=None, is_only_npu=False):
    kernel_name = json.loads(desc).get("op", "test")
    if os.environ.get("SYMBOLIC_TILING") == "1":
        kernel_name_str = "ST_TBE_" + kernel_name
    else:
        kernel_name_str = "AT_TBE_" + kernel_name

    logging.info("[KERNEL NAME]: %s", kernel_name_str)
    if is_only_npu:
        postfixs = [".o", ".json"]
        is_success = copy_to_akg_kernel_meta(kernel_name, postfixs)
        if not is_success:
            return None
        return kernel_name
    elif backend in ["cuda", "cpu", "aicore"]:
        mod = composite.build(desc, attrs, poly=poly)
        return mod

    else:
        raise ValueError("The current {} backend does not support.".format(backend))


def get_compare_result(desc, output, expect, output_indexes):
    # In profiling mode, mod_launch will return compute outputs and profiling value, only compute outputs needed here
    output = output if isinstance(output, (list, tuple)) else [output]
    expect = expect if isinstance(expect, (list, tuple)) else [expect]
    output = list(output)
    expect = list(expect)
    for i, _ in enumerate(expect):
        if expect[i].dtype == "complex128" or expect[i].dtype == "complex64":
            final_shape = functools.reduce(lambda x, y: x*y, output[i].shape)
            flattern_output = output[i].reshape((final_shape,))
            output_real = []
            output_imag = []
            for k, _ in enumerate(flattern_output):
                if k % 2 == 0:
                    output_real.append(flattern_output[k])
                else:
                    output_imag.append(flattern_output[k])
            output[i] = np.vectorize(complex)(output_real, output_imag)
            output[i] = output[i].reshape(expect[i].shape)
    if len(output) != len(expect):
        raise RuntimeError("output and expect have different length, {} vs {}".format(len(output), len(expect)))

    compare_tolerance = get_compare_tolerance(desc, output_indexes)
    compare_res = list(map(_compare_func, output, expect, compare_tolerance))
    return compare_res


def get_result(desc, poly, attrs=None, profiling=True, need_compare=True, precision_check=True, is_only_npu=False):
    backend = _get_backend(desc)
    mod = get_model(desc, poly, backend, attrs, is_only_npu)
    arch = get_ascend_type(_get_json_dict(desc))
    if mod is None:
        return False, None
    input_for_mod, expect, output_indexes = gen_json_data(desc, with_compute=precision_check)
    output = utils.mod_launch(mod, input_for_mod, output_indexes, arch=arch)

    if not precision_check:
        logging.info("No precision error check!")
        return True, None

    if not need_compare:
        return True, None

    cycles = None
    if isinstance(output, tuple) and len(output) > 0 and isinstance(output[-1], dict):
        cycles = output[1]
        output = output[0]

    compare_res = get_compare_result(desc, output, expect, output_indexes)

    if not all(compare_res):
        try:
            source = (mod.imported_modules[0] if backend == "cuda" else mod).get_source()
            logging.debug(source)
            _dump_info(desc, attrs, poly, input_for_mod, output, expect)
            logging.warning("Compare results: %s", str(compare_res))
        except UnboundLocalError:
            logging.error("Maybe you are using TBE for codegen and there is no mod. Please try `export USE_AKG_EMIT_ASCEND=AKG`")
        except Exception as e:
            logging.error("Unknown error: {}".format(e))
        logging.error("Compare Fail!")
        return False, None
    if profiling and backend in ["cuda", "cpu"]:
        ctx = tvm.context(backend, 0)
        has_complex = False
        for i in input_for_mod:
            if i.dtype == "complex64" or i.dtype == "complex128":
                has_complex = True
                break
        if has_complex == False:
            inputs = to_tvm_nd_array(input_for_mod, ctx)
            target_profiling(mod, *inputs, target=backend, repeat_time=1000)
    return True, cycles


@pytest.mark.skip
def test_single_file(input_file, attrs, poly, profiling=True, max_run_times=3):
    if not input_file.endswith(".info") and not input_file.endswith(".json"):
        print("Skip {}, only process file with .info or .json suffix".format(input_file))
        return
    logging.info("test file: %s", input_file)

    enable_input_cache()

    with open(input_file, 'r') as f:
        desc = f.read()
        for i in range(max_run_times):
            if get_profiling_compare_result(desc, poly, attrs, profiling):
                logging.info("Run Pass! max run time: %d, current run time: %d", max_run_times, i + 1)
                return
            logging.info("Precision error! max run time: %d, current run time: %d", max_run_times, i + 1)
        raise ValueError("Precision Error")


@pytest.mark.skip
def test_json_dir(poly, use_custom, json_dir="./json_dir/", online_tuning=0):
    # enable input cache for json dir testing
    enable_input_cache()

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
            logging.info("Begin run [No.%d/%d] file:%s", idx, len(files), input_file)
            idx = idx + 1
            desc = f.read()
            attrs = dims_dict.get(input_file, {}) if use_custom else {}
            need_compare = True
            if online_tuning:
                attrs["online_tuning"] = online_tuning
                need_compare = False
            if not get_profiling_compare_result(desc, poly, attrs, need_compare=need_compare):
                logging.info("----------Error Json name is----------")
                logging.info(input_file)
                raise ValueError("Precision Error")
    logging.info("All Json files run PASS!")


def get_op_cycles_info(cycle_info_file, old_op_cycles=100000000):
    with open(cycle_info_file, 'r') as f:
        op_cycles = int(f.read())
    diff = old_op_cycles - op_cycles
    return op_cycles, diff


def test_ci(profile=False, poly=False):
    pwd = os.path.dirname(os.path.abspath(__file__))
    ci_path = pwd + "/ascend_ci/"
    target_process = ["cuda", "aicore"]
    if profile:
        need_update = False
        base_json_file = pwd + "/ascend_ci/base.json"
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
                logging.info("------ Skip %s", fi)
                continue
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  fi: %s", fi)
            flag, _ = get_result(desc, poly)
            if not flag:
                logging.info("----------Error Json info is----------")
                logging.info(desc)
                raise ValueError("Precision Error")
            elif not profile:
                logging.info("Composite Json %s pass!", fi)
            else:
                old_op_cycles = old_dict[fi]
                op_cycles, diff = get_op_cycles_info(cycle_info_file, old_op_cycles)
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


def test_customop(subfolder, profile=False, poly=True):
    pwd = os.path.dirname(os.path.abspath(__file__))
    ci_path = pwd + "/customop_ci/" + subfolder + "/"
    target_process = ["aicore"]
    if profile:
        need_update = False
        base_json_file = pwd + "/customop_ci/" + subfolder + "/base.json"
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
                logging.info("------ Skip %s", fi)
                continue
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  fi: %s", fi)
            flag, _ = get_result(desc, poly, precision_check=False)
            if not flag:
                logging.info("----------Error Json info is----------")
                logging.info(desc)
                raise ValueError("Precision Error")
            elif not profile:
                logging.info("Composite Json %s pass!", fi)
            else:
                old_op_cycles = old_dict[fi]
                op_cycles, diff = get_op_cycles_info(cycle_info_file, old_op_cycles)
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

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ci_customop():
    test_customop("custom_intrin")
    test_customop("lu")
    test_customop("solve_triangular")

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ascend_composite_false():
    # As the current test framework does not support running in scenario
    # where the composite is false, test compilation only
    pwd = os.path.dirname(os.path.abspath(__file__))
    ci_path = pwd + "/ascend_composite_false/"
    files = os.listdir(ci_path)
    for fi in files:
        with open(ci_path + fi, 'r') as f:
            desc = f.read()
            res = compilewithjson(desc)
            if not res:
                logging.info("----------Error Json info is----------")
                logging.info(desc)
                raise ValueError("Compile Error!")
    logging.info("All ops compile success!")


def main(argv):
    import getopt
    # disable pylint too broad Exception
    # pylint: disable=W0703
    try:
        options, args = getopt.getopt(argv, "atldcf:mh", ["auto", "manual", "ci", "profile", "tune", "lower", "benchmark=",
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
            elif option in ("-l", "--lower"):
                attrs_list["is_tbe_codegen"] = True
            elif option in ("-bm", "--benchmark"):
                attrs_list["benchmark"] = True
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
    except Exception as e:
        print_usage()
        return


    if attrs_list.get("benchmark"):
        frontend_envdict = [
            {"SYMBOLIC_TILING": "0"},
            {"SYMBOLIC_TILING": "1"},
        ]
        file_names = []
        if single_file:
            file_names.append(file_name)
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
                    files = os.listdir(p)
                    for input_file in files:
                        file_names.append(p + "/" + input_file)
        for file_name in file_names:
            for frontend_choice in frontend_envdict:
                for fk, fv in frontend_choice.items():
                    os.environ[fk] = fv
                    logging.info("Frontend Choice: {}={}".format(fk, os.environ[fk]))
                    if attrs_list.get("is_tbe_codegen"):
                        attrs_list.pop("is_tbe_codegen")
                    attrs_copy = copy.deepcopy(attrs_list)
                    if 1:
                        p = Process(target=test_single_file,
                            args=(file_name, attrs_copy, poly, True,1))
                        p.start()
                        p.join()
                    else:
                        try:
                            test_single_file(file_name, attrs_copy, True, profiling=True, max_run_times=1)
                        except ValueError:
                            raise logging.error("Precision Error")
                        except Exception as e:
                            msg = str(e)
                            logging.error(msg)
    else:
        if single_file:
            test_single_file(file_name, attrs_list, poly, True)

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
                test_json_dir(poly, use_custom, online_tuning=online_tuning)

        elif ci_test:
            poly = False
            test_ci(use_profiling, poly)
        else:
            print_usage()


if __name__ == "__main__":
    main(sys.argv[1:])
