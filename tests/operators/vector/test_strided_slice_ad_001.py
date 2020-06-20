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
from test_run.strided_slice_ad_run import strided_slice_ad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_strided_slice_ad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("strided_slice_ad_001", strided_slice_ad_run, ([4, 4, 8, 8], [0, 0, 0, 0], [4, 4, 8, 8], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_002", strided_slice_ad_run, ([4, 4, 8, 8], [0, 0, 0, 0], [2, 2, 8, 8], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_003", strided_slice_ad_run, ([4, 4, 8, 8], [1, 1, 0, 0], [3, 3, 8, 8], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_004", strided_slice_ad_run, ([4, 8, 8], [1, 0, 0], [3, 8, 8], [1, 1, 1], "float16")),
            ("strided_slice_ad_005", strided_slice_ad_run, ([4, 4, 8, 8], [0, 1, 0, 0], [3, 4, 8, 8], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_006", strided_slice_ad_run, ([4, 4, 8, 8], [0, 0, 4, 0], [3, 4, 8, 8], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_007", strided_slice_ad_run, ([4, 4, 8, 8], [0, 1, 0, 0], [3, 4, 4, 8], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_008", strided_slice_ad_run, ([4, 4, 8, 8], [0, 0, 4, 0], [4, 4, 8, 8], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_009", strided_slice_ad_run, ([4, 4, 8, 8], [0, 0, 0, 0], [4, 4, 8, 4], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_010", strided_slice_ad_run, ([4, 8, 8], [0, 0, 0], [4, 8, 4], [1, 1, 1], "float16")),
            ("strided_slice_ad_011", strided_slice_ad_run, ([8, 8, 16, 16], [2, 4, 0, 0], [4, 8, 8, 16], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_012", strided_slice_ad_run, ([64, 16, 16], [0, 1, 0], [64, 2, 16], [1, 1, 1], "float16")),
            ("strided_slice_ad_013", strided_slice_ad_run, ([32, 32, 16], [0, 0, 0], [32, 1, 16], [1, 1, 1], "float16")),
            ("strided_slice_ad_014", strided_slice_ad_run, ([8, 8, 16, 16], [0, 0, 0, 0], [4, 8, 8, 16], [1, 1, 1, 1], "float16")),
            ("strided_slice_ad_015", strided_slice_ad_run, ([4, 8, 8], [0, 0, 0], [4, 8, 8], [1, 1, 1], "float16")),
            ("strided_slice_ad_016", strided_slice_ad_run, ([4, 8, 8], [0, 0, 0], [4, 8, 8], [1, 1, 1], "float16")),
            ("strided_slice_ad_017", strided_slice_ad_run, ([4, 16, 16], [0, 0, 0], [4, 16, 16], [1, 2, 1], "float16")),
            ("strided_slice_ad_018", strided_slice_ad_run, ([4, 16, 16], [0, 0, 0], [4, 16, 16], [2, 1, 1], "float16")),
            ("strided_slice_ad_019", strided_slice_ad_run, ([4, 16, 16], [0, 0, 0], [4, 16, 16], [2, 2, 1], "float16")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("strided_slice_ad_001", strided_slice_ad_run, ([4, 4, 8, 8], [0, 0, 0, 0], [4, 4, 8, 8], [1, 1, 1, 1], "float32"), [(1, 0)]),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
