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
from base import TestBase
import pytest


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_logsoftmax_grad"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("logsoftmax_grad_001", "logsoftmax_grad_run", ((64, 2), "float16", -1, "cce_logsoftmax_grad_fp16")),
            # ("logsoftmax_grad_002", "logsoftmax_grad_run", ((20, 32000), "float16", -1, "cce_logsoftmax_fp16")),
            # ("logsoftmax_grad_003", "logsoftmax_grad_run", ((160, 30522), "float16", -1, "cce_logsoftmax_fp16")),
            # ("logsoftmax_grad_004", "logsoftmax_grad_run", ((1280, 21128), "float16", -1, "cce_logsoftmax_fp16")),
            # ("logsoftmax_grad_005", "logsoftmax_grad_run", ((1280, 30522), "float16", -1, "cce_logsoftmax_fp16")),

        ]
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("logsoftmax_grad_002", "logsoftmax_grad_run", ((8, 2), "float16", -1, "cce_logsoftmax_fp16")),
        ]

        self.testarg_rpc_cloud = [
            # float:[64, 2] = float:[64, 2]
            ("logsoftmax_grad_001", "logsoftmax_grad_run", ((64, 2), "float16", -1, "cce_logsoftmax_fp16")),
            # float:[160, 30522] = float:[160, 30522]
            ("logsoftmax_grad_002", "logsoftmax_grad_run", ((160, 30522), "float32", -1, "cce_logsoftmax_fp32")),
            # float:[8, 2] = float:[8, 2]
            ("logsoftmax_grad_003", "logsoftmax_grad_run", ((8, 2), "float16", -1, "cce_logsoftmax_fp16")),
            # float:[1280, 30522] = float:[1280, 30522]
            ("logsoftmax_grad_004", "logsoftmax_grad_run", ((1280, 30522), "float32", -1, "cce_logsoftmax_fp32")),
            ("logsoftmax_grad_002", "logsoftmax_grad_run", ((20, 32000), "float32", -1, "cce_logsoftmax_fp32")),
            ("logsoftmax_grad_003", "logsoftmax_grad_run", ((1280, 21128), "float32", -1, "cce_logsoftmax_fp32")),

        ]
        self.testarg_level1 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("logsoftmax_grad_001", "logsoftmax_grad_run", ((64, 2), "float16", -1, "cce_logsoftmax_grad_fp16")),
            ("logsoftmax_grad_002", "logsoftmax_grad_run", ((8, 2), "float16", -1, "cce_logsoftmax_grad_fp16")),

        ]
        self.testarg_cloud_level1 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            #("logsoftmax_grad_002", "logsoftmax_grad_run", ((8, 2), "float16", -1, "cce_logsoftmax_fp16"), ((8, 8), (94, 94))),
        ]

        self.testarg_rpc_cloud_level1 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("logsoftmax_grad_001", "logsoftmax_grad_run", ((64, 2), "float32", -1, "cce_logsoftmax_grad_fp32")),
            ("logsoftmax_grad_002", "logsoftmax_grad_run", ((8, 2), "float32", -1, "cce_logsoftmax_grad_fp32")),

            ("logsoftmax_grad_003", "logsoftmax_grad_run", ((160, 30522), "float32", -1, "cce_logsoftmax_fp32")),
            ("logsoftmax_grad_004", "logsoftmax_grad_run", ((1280, 30522), "float32", -1, "cce_logsoftmax_fp32")),

            ("logsoftmax_grad_005", "logsoftmax_grad_run", ((8, 16), "float32", -1, "cce_logsoftmax_fp32")),
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

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[0]])

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud_level1(self):
        self.common_run(self.testarg_cloud_level1)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud_level1(self):
        self.common_run(self.testarg_rpc_cloud_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
