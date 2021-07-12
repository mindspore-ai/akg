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
from tests.common.test_run.relu_grad_run import relu_grad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_relu_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("relu_grad_001", relu_grad_run, ((1, 128), "float16"), ((16, 0), (1, 0))),
            #("relu_grad_002", relu_grad_run, ((4,129,129,256), "float16")),
            ("relu_grad_003", relu_grad_run, ((4, 129, 129, 48), "float16")),
            ("relu_grad_004", relu_grad_run, ((4, 257, 257, 128), "float16")),
            ("relu_grad_005", relu_grad_run, ((4, 257, 257, 64), "float16")),
            ("relu_grad_006", relu_grad_run, ((4, 65, 65, 728), "float16"), ((1, 1), (1, 1), (1, 1), (728, 1))),
            ("relu_grad_007", relu_grad_run, ((4, 33, 33, 256), "float16")),
            ("relu_grad_008", relu_grad_run, ((4, 33, 33, 2048), "float16")),
            ("relu_grad_009", relu_grad_run, ((4, 33, 33, 1024), "float16")),
            ("relu_grad_010", relu_grad_run, ((4, 33, 33, 728), "float16"), ((1, 1), (1, 1), (1, 1), (768, 1))),
            ("relu_grad_011", relu_grad_run, ((4, 1, 1, 256), "float16")),
            ("relu_grad_012", relu_grad_run, ((4, 129, 129, 128), "float16")),
            ("relu_grad_013", relu_grad_run, ((4, 257, 257, 32), "float16")),
            ("relu_grad_014", relu_grad_run, ((4, 129, 129, 304), "float16")),
            ("relu_grad_015", relu_grad_run, ((4, 33, 33, 1536), "float16")),
            ("relu_grad_016", relu_grad_run, ((4, 65, 65, 256), "float16")),
        ]
        self.testlenet_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("relu_grad_001", relu_grad_run, ((1, 16, 7, 7), "float16")),
            ("relu_grad_002", relu_grad_run, ((1, 6, 15, 15), "float16")),

        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        # self.common_run(self.testarg_rpc_cloud)
        self.common_run(self.testlenet_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
