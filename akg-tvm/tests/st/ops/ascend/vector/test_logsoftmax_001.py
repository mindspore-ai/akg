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
        case_name = "test_akg_logsoftmax"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("logsoftmax_01", "logsoftmax_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
            ("logsoftmax_02", "logsoftmax_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),
            ("logsoftmax_03", "logsoftmax_run", ((64, 2), "float16", -1, "cce_logsoftmax_fp16")),
            ("logsoftmax_04", "logsoftmax_run", ((8, 2), "float16", -1, "cce_logsoftmax_fp16")),
            ("logsoftmax_05", "logsoftmax_run", ((64, 2), "float16", 0, "cce_logsoftmax_fp16")),
            ("logsoftmax_06", "logsoftmax_run", ((8, 2), "float16", 0, "cce_logsoftmax_fp16")),
            # ("logsoftmax_07", "logsoftmax_run", ((160, 30522), "float16", -1, "cce_logsoftmax_fp16"), ((1, 1), (30522,30522))),
            # ("logsoftmax_08", "logsoftmax_run", ((1280, 30522), "float16", -1, "cce_logsoftmax_fp16"), ((1, 1), (30522,30522))),

        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("logsoftmax_02", "logsoftmax_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),
        ]

        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("logsoftmax_01_fp32", "logsoftmax_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
            ("logsoftmax_02_fp32", "logsoftmax_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),
            ("logsoftmax_03_fp32", "logsoftmax_run", ((160, 30522), "float32", -1, "cce_logsoftmax_fp32"), ((1, 1), (30522,30522))),
            ("logsoftmax_04_fp32", "logsoftmax_run", ((1280, 30522), "float32", 1, "cce_logsoftmax_fp32"), ((1, 1), (30522,30522))),
            ("logsoftmax_05_fp32", "logsoftmax_run", ((1280, 30522), "float32", -1, "cce_logsoftmax_fp32"), ((1, 1), (30522,30522))),
            ("logsoftmax_06_fp32", "logsoftmax_run", ((20, 32000), "float32", 1, "cce_logsoftmax_fp32")),
            ("logsoftmax_07_fp32", "logsoftmax_run", ((1, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[64, 2] = float:[64, 2]
            ("logsoftmax_001_fp32", "logsoftmax_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[8, 2] = float:[8, 2]
            ("logsoftmax_003_fp32", "logsoftmax_run", ((8, 2), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[1280, 21128] = float:[1280, 21128]
            ("logsoftmax_005_fp32", "logsoftmax_run", ((1280, 21128), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[64, 2] = float:[64, 2]
            ("logsoftmax_006_fp32", "logsoftmax_run", ((64, 2), "float32", -1, "cce_logsoftmax_fp32")),
        ]
        self.testarg_level1 = [
            # testflag,opfuncname,testRunArgs, dimArgs
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
