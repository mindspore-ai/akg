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
        case_name = "test_gelu_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_ci = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype
            ("004_gelu_input_1280_1024", "gelu_run", ((1280, 1024), "float16")),
            ("008_gelu_input_4928_1024", "gelu_run", ((4928, 1024), "float16")),
        ]

        self.testarg_nightly = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype
            ("005_gelu_input_8192_4096", "gelu_run", ((8192, 4096), "float16")),
            ("006_gelu_input_8192_1024", "gelu_run", ((8192, 1024), "float16")),
            ("007_gelu_input_32768_1024", "gelu_run", ((32768, 1024), "float16")),
            ("009_gelu_input_32768_4096", "gelu_run", ((32768, 4096), "float16")),
        ]

        self.testarg_rpc_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype
            #  float16:[64 * 128, 4096] = float16:[64 * 128, 4096]
            ("gelu_001_input_8192_4096", "gelu_run", ((64 * 128, 4096), "float32")),
            # float16:[64 * 20, 1024] = float:[64 * 20, 1024]
            ("gelu_002_input_1280_1024", "gelu_run", ((1280, 1024), "float32")),
            ("gelu_008", "gelu_run", ((40, 768), "float32")),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_ci(self):
        self.common_run(self.testarg_ci)

    def test_run_nightly(self):
        self.common_run(self.testarg_nightly)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
