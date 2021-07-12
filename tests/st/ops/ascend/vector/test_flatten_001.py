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
        case_name = "test_flatten_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_ci = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype
            ("001_flatten_input", "flatten_run", ((16, 16), "float16")),
            ("002_flatten_input", "flatten_run", ((16, 16), "float32")),
            ("003_flatten_input", "flatten_run", ((16, 16), "int8")),
            ("004_flatten_input", "flatten_run", ((16, 16), "int16")),
            ("005_flatten_input", "flatten_run", ((16, 16), "int32")),
            ("006_flatten_input", "flatten_run", ((16, 16), "int64")),
            ("007_flatten_input", "flatten_run", ((16, 16), "uint8")),
            ("008_flatten_input", "flatten_run", ((16, 16), "uint16")),
            ("009_flatten_input", "flatten_run", ((16, 16), "uint32")),
            ("010_flatten_input", "flatten_run", ((16, 16), "uint64")),
        ]

        self.testarg_nightly = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype
        ]

        self.testarg_rpc_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype
        ]

        return

    @pytest.mark.level2
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
