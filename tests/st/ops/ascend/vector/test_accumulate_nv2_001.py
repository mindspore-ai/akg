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

import os
from tests.common.base import TestBase


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_accumulate_nv2_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_level1 = [
            # testcase args: (shape, dtype, tensor_num)
            ("accumulate_nv2_001", "accumulate_nv2_run", ((16, 16), "float16", 4)),
            ("accumulate_nv2_002", "accumulate_nv2_run", ((16, 16), "float32", 4)),
            ("accumulate_nv2_003", "accumulate_nv2_run", ((16, 16), "int32", 4)),
            ("accumulate_nv2_006", "accumulate_nv2_run", ((16, 16), "float16", 1)),
            ("accumulate_nv2_007", "accumulate_nv2_run", ((16, 16), "float32", 1)),
            ("accumulate_nv2_008", "accumulate_nv2_run", ((16, 16), "int32", 1)),
        ]
        self.testarg_level2 = [
            ("accumulate_nv2_004", "accumulate_nv2_run", ((16, 16), "int8", 4)),
            ("accumulate_nv2_005", "accumulate_nv2_run", ((16, 16), "uint8", 4)),
            ("accumulate_nv2_009", "accumulate_nv2_run", ((16, 16), "int8", 1)),
            ("accumulate_nv2_010", "accumulate_nv2_run", ((16, 16), "uint8", 1)),
        ]

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
