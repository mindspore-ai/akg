# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
import pytest
import os
import logging
from akg import composite
from akg.utils import kernel_exec as utils
from akg.utils.composite_op_helper import gen_json_data
from tests.common.tensorio import compare_tensor

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ms_composite_buffer_stitch():
    pwd = os.path.dirname(os.path.abspath(__file__))
    ci_path = os.path.join(pwd, "stitch_cases")
    test_composite_stitch(ci_path)
    return True

@pytest.mark.skip
def test_composite_stitch(ci_path):
    files = os.listdir(ci_path)
    flag = True
    for fi in files:
        with open(os.path.join(ci_path, fi), 'r') as f:
            print("\033[94m%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%file: \033[0m", fi)
            desc = f.read()
        poly = True
        attrs = {}
        reduce_lib_key = "enable_akg_reduce_lib"
        attrs[reduce_lib_key] = poly
        mod = composite.build(desc, attrs, poly=poly)
        rtol = 0.001
        atol = 0.005
        max_run_times = 3
        case_flag = False

        for i in range(max_run_times):
            input_for_mod, expect, output_indexes = gen_json_data(desc)
            output = utils.mod_launch(mod, input_for_mod, output_indexes)
            if len(output_indexes) > 1:
                if all(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), output, expect)):
                    case_flag = True
                    break
            else:
                if compare_tensor(output, expect, rtol=rtol, atol=atol):
                    case_flag = True
                    break
        if not case_flag:
            logging.info("\033[91mComposite Json {} fail!\033[0m".format(fi))
        else:
            logging.info("\033[92mComposite Json {} pass!\033[0m".format(fi))
        flag &= case_flag
    if not flag:
        raise ValueError("Precision Error")
    logging.info("All ops are ok!")