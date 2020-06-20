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
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.bitwise_not_run import bitwise_not_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_bitwise_not_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_bitwise_not", bitwise_not_run, ((12,), "int32", "bitwise_not")),
            ("001_bitwise_not", bitwise_not_run, ((12, 15), "int32", "bitwise_not")),
            ("001_bitwise_not", bitwise_not_run, ((12, 15), "int8", "bitwise_not")),
        ]

        self.testarg_cloud = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_bitwise_not", bitwise_not_run, ((512,), "int32", "bitwise_not")),
            ("001_bitwise_not", bitwise_not_run, ((32, 16), "int32", "bitwise_not")),
            ("001_bitwise_not", bitwise_not_run, ((32, 16), "int8", "bitwise_not")),
        ]
        self.testarg_level1 = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_bitwise_not", bitwise_not_run, ((256,), "int32", "bitwise_not")),
            ("001_bitwise_not", bitwise_not_run, ((32, 32), "int32", "bitwise_not")),
            ("001_bitwise_not", bitwise_not_run, ((32, 32), "int8", "bitwise_not")),
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

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
