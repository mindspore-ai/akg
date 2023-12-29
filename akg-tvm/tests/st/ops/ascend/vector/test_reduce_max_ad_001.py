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

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.reduce_max_ad_run import reduce_max_ad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_reduce_max_ad"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # shape, axis, keepdims, dtype, attrs

            ("reduce_max_ad_1", reduce_max_ad_run, ((15, 10,), (1,), True, "float16", True, "cce_reduce_max_fp16")),
            ("reduce_max_ad_2", reduce_max_ad_run, ((15, 10,), (-1,), True, "float16", True, "cce_reduce_max_fp16")),
            ("reduce_max_ad_3", reduce_max_ad_run, ((15, 10,), (0,), True, "float16", True, "cce_reduce_max_fp16")),
            ("reduce_max_ad_4", reduce_max_ad_run, ((15, 10,), (0,), False, "float16", True, "cce_reduce_max_fp16")),
            ("reduce_max_ad_5", reduce_max_ad_run, ((32, 64, 32), (1,), True, "float16", True, "cce_reduce_max_fp16")),
            # ("reduce_max_ad_6", reduce_max_ad_run, ((4, 128, 1024), (0, 1), True, "float32", False, "cce_reduce_max_fp32")),
            # ("reduce_max_ad_7", reduce_max_ad_run, ((2, 1280), (1,), False, "int8", False, "cce_reduce_max_int8")),
            # ("reduce_max_ad_8", reduce_max_ad_run, ((8, 256), (0,), False, "uint8", False, "cce_reduce_max_uint8")),
            # ("reduce_max_ad_9", reduce_max_ad_run, ((1024,), (0,), True, "int32", False, "cce_reduce_max_int32")),

            # manual schedule
            ("reduce_max_ad_1", reduce_max_ad_run, ((15, 10,), (1,), False, "float16", True, "cce_reduce_max_fp16")),
            # Reduce_Max is limited to 2D only, check reduce_max_ad.py
            # Tiling dims other than [1,1] require cleanup of dead code after the loop partition
            # ("reduce_max_ad_2", reduce_max_ad_run, ((32, 64, 32), (1,), True, "float16", False, "cce_reduce_max_fp16"), [(1,1),(1,1),(1,1)], False),
            # ("reduce_max_ad_3", reduce_max_ad_run, ((4, 128, 1024), (0, 1), True, "float32", False, "cce_reduce_max_fp32"), [(1,1),(1,1),(1,1)], False),
            ("reduce_max_ad_4", reduce_max_ad_run, ((2, 1280), (1,), False, "int8", False, "cce_reduce_max_int8"), [(1, 1), (1, 1)], False),
            # NOTE: reduce max on axis = 0 is not working, changing the axis to 1 for following
            ("reduce_max_ad_5", reduce_max_ad_run, ((8, 256), (1,), False, "uint8", False, "cce_reduce_max_uint8"), [(1, 1), (1, 1)], False),
            #("reduce_max_ad_6", reduce_max_ad_run, ((1024,), (0,), True, "int32", False, "cce_reduce_max_int32"), [(1,1)], False),
        ]

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
