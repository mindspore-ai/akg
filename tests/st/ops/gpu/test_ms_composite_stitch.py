# Copyright 2021 Huawei Technologies Co., Ltd
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
import logging
from akg import composite
from akg.utils import kernel_exec as utils
from tests.common.gen_json_data import gen_json_data
from tests.common.base import get_rtol_atol
from tests.common.tensorio import compare_tensor


def test_composite_stitch(ci_path):
    files = os.listdir(ci_path)
    for fi in files:
        with open(ci_path + fi, 'r') as f:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%file: ", fi)
            desc = f.read()
            poly = True
            attrs = {}
            reduce_lib_key = "enable_akg_reduce_lib"
            attrs[reduce_lib_key] = poly
            mod = composite.build(desc, attrs, poly=poly)
            input_for_mod, expect, output_indexes = gen_json_data(desc)
    output = utils.mod_launch(mod, input_for_mod, output_indexes)

    rtol, atol = get_rtol_atol("FUSED", "float32")
    flag = True
    if len(output_indexes) > 1:
        if not all(map(lambda x, y: compare_tensor(x, y, rtol=rtol, atol=atol), output, expect)):
            logging.info(mod.imported_modules[0].get_source())
            flag = False
    else:
        if not compare_tensor(output, expect, rtol=rtol, atol=atol):
            logging.info(mod.imported_modules[0].get_source())
            flag = False
            if not flag:
                logging.info("----------Error Json info is----------")
                logging.info(desc)
                raise ValueError("Precision Error")
            else:
                logging.info("Composite Json {} pass!".format(fi))
            assert (flag)
    logging.info("All ops are ok!")
