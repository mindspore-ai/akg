# Copyright 2020 Huawei Technologies Co., Ltd
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

import sys
import os
import json
import pytest
import logging
from akg import composite
from akg.utils import custom_tiling
from akg.utils import kernel_exec as utils
from gen_json_data import gen_json_data
from base import get_rtol_atol
from tensorio import compare_tensor
logging.getLogger().setLevel(logging.INFO)

def print_usage():
    logging.info("Usage: test_composite_json.py <JSON_FILE> to run single file.")
    logging.info("Usage: test_composite_json.py -d to run files in a directory, default to be ./json_dir.")
    logging.info("Usage: test_composite_json.py -ci to run ci files.")
    logging.info("compile composite op")

def get_result(desc, attrs=None):
    input_for_mod, expect, output_indexes = gen_json_data(desc)

    if attrs:
        mod = composite.build(desc, attrs)
    else:
        mod = composite.build(desc)
    output = utils.mod_launch(mod, input_for_mod, output_indexes)

    rtol, atol = get_rtol_atol("FUSED", "float32")
    flag = True
    if len(output_indexes) > 1:
        if not all(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), output, expect)):
            flag = False
    else:
        if not compare_tensor(output, expect, rtol=rtol, atol=atol):
            flag = False
    return flag

@pytest.mark.skip
def test_single_file(input_file, use_custom):
    with open(input_file, 'r') as f:
        desc = f.read()
        if use_custom:
            attrs = {}
            attrs["dim"] = custom_tiling.set_dims(((4, 1), (4, 1)))
            flag = get_result(desc, attrs)
        else:
            flag = get_result(desc)
        if flag:
            logging.info("Run Pass!")
        else:
            logging.info("Precision Error")

@pytest.mark.skip
def test_json_dir():
    json_dir = "./json_dir/"
    json_dims_file = "./json_dir/dims.json"
    files = os.listdir(json_dir)
    flag = True
    with open(json_dims_file, 'r') as f:
        base = f.read()
        dims_dict = json.loads(base)
    for input_file in files:
        with open(json_dir + input_file, 'r') as f:
            if input_file == "dims.json":
                continue
            desc = f.read()
            if input_file in dims_dict:
                dim_info = dims_dict[input_file]
                attrs = {'dim': dim_info}
                flag = get_result(desc, attrs)
            else:
                flag = get_result(desc)
            if not flag:
                logging.info("----------Error Json name is----------")
                logging.info(input_file)
                raise ValueError("Precision Error")
    logging.info("All Json files run PASS!")

def get_op_cycles_info(desc, cycle_info_file, old_op_cycles=100000000):
    with open(cycle_info_file, 'r') as f:
        op_cycles = int(f.read())
    diff = old_op_cycles - op_cycles
    return op_cycles, diff

@pytest.mark.level0
def test_ci(profile=False):
    ci_path = "./need_adapt/"
    if profile:
        need_update = False
        base_json_file = "./need_adapt/base.json"
        cycle_info_file = "./cycle_path/a.txt"
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
            flag = get_result(desc)
            if not flag:
                logging.info("----------Error Json info is----------")
                logging.info(desc)
                raise ValueError("Precision Error")
            elif not profile:
                logging.info("Composite Json {} pass!".format(fi))
            else:
                old_op_cycles = old_dict[fi]
                op_cycles, diff = get_op_cycles_info(desc, cycle_info_file, old_op_cycles)
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
            assert(flag)
    logging.info("All ops are ok!")
    if profile:
        if need_update:
            logging.info("Need to Update Baseline!!!")
            with open(base_json_file, 'w', encoding='utf-8') as f:
                json.dump(old_dict, f, indent=4)
        else:
            logging.info("No significant performance improvement. Do not need to update Baseline!")

def main(argv):
    if len(argv) in [1, 2] and (argv[0].endswith(".info") or argv[0].endswith(".json")):
        use_custom = len(argv) == 2 and argv[1] == 'c'
        test_single_file(argv[0], use_custom)
    elif len(argv) == 1 and argv[0] == "-d":
        test_json_dir()
    elif len(argv) == 1 and argv[0] == "-ci":
        test_ci(profile=False)
    elif len(argv) == 1 and argv[0] == "-cip":
        test_ci(profile=True)
    else:
        print_usage()

if __name__ == "__main__":
    main(sys.argv[1:])
