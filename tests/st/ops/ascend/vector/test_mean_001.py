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


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_mean_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # set dim error

#            ("mean_011", "mean_run", ((10, 3, 10, 9, 8, 5, 4), 'float16', (5, 4, 1, 2), False, 'mean'), ),

            # ("mean_011", "mean_run", ((10, 3, 10, 9, 8, 5, 4), 'float16', (-2, -3, -6, -5), False, 'cce_mean_1_8_fp16'), ),
            ("mean_01", "mean_run", ((8,), "float16", (0,), False, "cce_mean_1_64_fp16")),
            ("mean_01", "mean_run", ((8,), "float32", (0,), False, "cce_mean_1_64_fp32")),

            ("mean_011", "mean_run", ((8, 3), 'float16', (0,), False, 'cce_mean_1_8_fp16'), ),
            ("mean_012", "mean_run", ((9, 1, 2, 4, 8, 2, 8), 'float16', (5,), False, 'cce_mean_1_8_fp16'),),
            ("mean_01", "mean_run", ((64,), "float16", (0,), False, "cce_mean_1_64_fp16")),
            ("mean_02", "mean_run", ((8,), "float16", (0,), True, "cce_mean_1_8_fp16")),
            ("mean_03", "mean_run", ((64, 128, 1024), "float16", (2,), False, "cce_mean_64_128_1024_fp16"),),
            ("mean_04", "mean_run", ((1024, 1024), "float16", (1,), False, "cce_mean_fp16"),),
            ("mean_05", "mean_run", ((1280, 1024), "float16", (1,), False, "cce_mean_fp16"),),
            ("mean_06", "mean_run", ((8192, 1024), "float16", (1,), False, "cce_mean_fp16"),),
            ("mean_07", "mean_run", ((160, 1024), "float16", (1,), True, "cce_mean_fp16"),),
            ("mean_08", "mean_run", ((8, 128, 1024), "float16", (2,), True, "cce_mean_64_128_1024_fp16"),),
            # resnet 50 5D:
            ("mean_019", "mean_run", ((32, 128, 7, 7, 16), "float16", (2, 3), True, "cce_mean_32_128_7_7_16"))
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("mean_01", "mean_run", ((64,), "float32", (0,), False, "cce_mean_1_64_fp16")),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("mean_03_fp32", "mean_run", ((64, 128, 1024), "float32", (2,), False, "cce_mean_64_128_1024_fp32")),
            ("mean_02_fp32", "mean_run", ((8,), "float32", (0,), True, "cce_mean_1_8_fp32")),
            ("mean_04_fp32", "mean_run", ((1024, 1024), "float32", (1,), False, "cce_mean_fp32")),
            ("mean_01_fp32", "mean_run", ((64,), "float32", (0,), False, "cce_mean_1_64_fp32")),
            ("mean_08_fp32", "mean_run", ((8, 128, 1024), "float32", (2,), True, "cce_mean_64_128_1024_fp32")),
            ("mean_05_fp32", "mean_run", ((1280, 1024), "float32", (1,), False, "cce_mean_fp32")),
            ("mean_06_fp32", "mean_run", ((8192, 1024), "float32", (1,), False, "cce_mean_fp32")),
            ("mean_07_fp32", "mean_run", ((160, 1024), "float32", (1,), True, "cce_mean_fp32")),
            # float - int32:[64, 128, 1024] - [1]
            ("mean_001", "mean_run", ((64, 128, 1024), "float32", (0,), False, "cce_mean_64_128_1024_fp32")),
            # float - int32:[8] - [1]
            ("mean_002", "mean_run", ((8,), "float32", (0,), True, "cce_mean_8_fp32")),
            # float - int32:[1024, 1024] - [1]
            ("mean_003", "mean_run", ((1024, 1024), "float32", (0,), False, "cce_mean_1024_1024_fp32")),
            # float - int32:[64] - [1]
            ("mean_004", "mean_run", ((64,), "float32", (0,), False, "cce_mean_64_fp32")),
            # float - int32:[8, 128, 1024] - [1]
            ("mean_005", "mean_run", ((8, 128, 1024), "float32", (0, 1, 2), True, "cce_mean_64_128_1024_fp32")),
            # float - int32:[1280, 1024] - [1]
            ("mean_006", "mean_run", ((1280, 1024), "float32", (0,), False, "cce_mean_1280_1024_fp32")),
            # float - int32:[8192, 1024] - [1]
            ("mean_007", "mean_run", ((8192, 1024), "float32", (0, 1), False, "cce_mean_8192_1024_fp32")),
            # float - int32:[160, 1024] - [1]
            ("mean_008", "mean_run", ((160, 1024), "float32", (0, 1), True, "cce_mean_160_1024_fp32")),
            # float - int32:[64, 128, 768] - [1] = float:[64, 128, 1]
            ("mean_009", "mean_run", ((64, 128, 768), "float32", (0, 1, 2), True, "cce_mean_64_128_1024_fp32")),
            # float - int32:[1280, 768] - [1] = float:[1280, 1]
            ("mean_010", "mean_run", ((1280, 768), "float32", (0, 1), True, "cce_mean_64_128_1024_fp32")),
            # float - int32:[8192, 768] - [1] = float:[8192, 1]
            ("mean_011", "mean_run", ((8192, 768), "float32", (0, 1), True, "cce_mean_64_128_1024_fp32")),
            # float - int32:[64] - [1] = float:[]
            ("mean_012", "mean_run", ((64,), "float32", (0,), False, "cce_mean_64_128_1024_fp32")),
            # resnet 50 5D:
            ("mean_019", "mean_run", ((32, 128, 7, 7, 16), "float32", (2, 3), True, "cce_mean"))
        ]

        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

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
