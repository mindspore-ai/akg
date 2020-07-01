# Copyright 2019 Huawei Technologies Co., Ltd
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

"""
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import datetime
import os

from base import TestBase
import pytest
from test_run.resize_bilinear_run import resize_bilinear_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_resize_bilinear_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # ("resize_bilinear_00", resize_bilinear_run, ([1, 8, 8, 16], [16, 16], "float16", "cce_resize_bilinear_fp16"),((16,1),(38,1),(53,1),(60,1))),
            # ("resize_bilinear_01", resize_bilinear_run, ([1,375,500,3], [468,625], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_03", resize_bilinear_run, ([1,500,333,3], [750,499], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_04", resize_bilinear_run, ([1,357,500,3], [624,875], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_05", resize_bilinear_run, ([1,385,500,3], [481,625], "float16", "cce_resize_bilinear_fp16")),   # setdim
            ("resize_bilinear_07", resize_bilinear_run, ([1, 9, 9, 3], [9, 9], "float16", "cce_resize_bilinear_fp16")),
            # ("resize_bilinear_08", resize_bilinear_run, ([1,339,500,3], [254,375], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_09", resize_bilinear_run, ([1,500,333,3], [375,249], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_10", resize_bilinear_run, ([1,1,1,256], [33,33], "float16", "cce_resize_bilinear_fp16")),       # setdim
            # ("resize_bilinear_11", resize_bilinear_run, ([1,335,500,3], [418,625], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_12", resize_bilinear_run, ([1,333,500,3], [166,250], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_13", resize_bilinear_run, ([1,500,375,3], [875,656], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_14", resize_bilinear_run, ([1,129,129,21], [513,513], "float16", "cce_resize_bilinear_fp16")),  # poly
            # ("resize_bilinear_15", resize_bilinear_run, ([1,357,500,3], [267,375], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_16", resize_bilinear_run, ([1,500,375,3], [625,468], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_17", resize_bilinear_run, ([1,500,313,3], [1000,626], "float16", "cce_resize_bilinear_fp16")),  # poly
            # ("resize_bilinear_18", resize_bilinear_run, ([1,338,500,3], [591,875], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_19", resize_bilinear_run, ([1,375,500,3], [656,875], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_20", resize_bilinear_run, ([1,415,500,3], [830,1000], "float16", "cce_resize_bilinear_fp16")),  # poly
            # ("resize_bilinear_21", resize_bilinear_run, ([1,375,500,3], [562,750], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_22", resize_bilinear_run, ([1,375,500,3], [281,375], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_23", resize_bilinear_run, ([1,500,281,3], [250,140], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_26", resize_bilinear_run, ([1,500,375,3], [250,187], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_27", resize_bilinear_run, ([1,500,378,3], [375,283], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_28", resize_bilinear_run, ([1,500,500,3], [375,375], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_29", resize_bilinear_run, ([1,418,500,3], [627,750], "float16", "cce_resize_bilinear_fp16")),   # poly
            # ("resize_bilinear_30", resize_bilinear_run, ([1,318,500,3], [397,625], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_31", resize_bilinear_run, ([1,112,500,3], [196,875], "float16", "cce_resize_bilinear_fp16")),   # Floating point exception (poly)
            # ("resize_bilinear_32", resize_bilinear_run, ([1,333,500,3], [416,625], "float16", "cce_resize_bilinear_fp16")),   # setdim
            # ("resize_bilinear_33", resize_bilinear_run, ([1,500,375,3], [1000,750], "float16", "cce_resize_bilinear_fp16")),  # poly
        ]

        self.test_level1 = [
            ("resize_bilinear_02", resize_bilinear_run, ([4, 129, 129, 21], [129, 129], "float16", "cce_resize_bilinear_fp16")),
            ("resize_bilinear_06", resize_bilinear_run, ([1, 129, 129, 48], [129, 129], "float16", "cce_resize_bilinear_fp16")),
            ("resize_bilinear_24", resize_bilinear_run, ([1, 400, 500, 3], [400, 500], "float16", "cce_resize_bilinear_fp16")),
            ("resize_bilinear_25", resize_bilinear_run, ([1, 333, 500, 3], [333, 500], "float16", "cce_resize_bilinear_fp16")),
            ("resize_bilinear_34", resize_bilinear_run, ([1, 9, 9, 48], [9, 9], "float16", "cce_resize_bilinear_fp16")),
            ("resize_bilinear_35", resize_bilinear_run, ([1, 375, 500, 3], [375, 500], "float16", "cce_resize_bilinear_fp16")),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.rpc_mini
    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_daily_ci(self):
        self.common_run(self.test_level1)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
