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
        case_name = "test_akg_softmax"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("softmax_01", "softmax_run", ((4, 4), "float16", -1, "cce_softmax_fp16")),
            ("softmax_02", "softmax_run", ((1, 1, 128, 128), "float16", -1, "cce_softmax_fp16")),
            ("softmax_03", "softmax_run", ((64, 16, 128, 128), "float16", -1, "cce_softmax_fp16")),
            # Test single 1d pattern
            ("softmax_04", "softmax_run", ((1547, 1220), "float32", 0, "cce_softmax_fp16")),
            ("softmax_05", "softmax_run", ((175, 855), "float32", 0, "cce_softmax_fp16")),
            ("softmax_06", "softmax_run", ((1, ), "float16", -1, "cce_softmax_fp16")),

        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs

            # float:[64, 16, 128, 128] = float:[64, 16, 128, 128]
            ("softmax_001", "softmax_run", ((64, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            # float:[8, 16, 128, 128] = float:[8, 16, 128, 128]
            ("softmax_002", "softmax_run", ((8, 16, 128, 128), "float32", -1, "cce_softmax_fp32")),
            ("softmax_003", "softmax_run", ((32, 10), "float32", -1, "cce_softmax_fp32")),
            ("softmax_004", "softmax_run", ((32, 10), "float16", -1, "cce_softmax_fp16")),

        ]
        self.testarg_level = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # ("softmax_001", "softmax_run", ((64, 16, 128, 128), "float16", -1, "cce_softmax_fp16")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[0]])

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return

# a = TestCase()
# a.setup()
# a.test_run()
