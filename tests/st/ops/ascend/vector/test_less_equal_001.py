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
        case_name = "test_auto_less_equal_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("003_less_equal", "less_equal_run", (((1,), (1,)), "float32", "less_equal")),
            ("003_less_equal", "less_equal_run", (((1,), (1,)), "int32", "less_equal")),
            ("001_less_equal", "less_equal_run", (((128,), (128,)), "float16", "less_equal")),
            ("002_less_equal", "less_equal_run", (((128, 128), (128, 128)), "float32", "less_equal")),
            ("003_less_equal", "less_equal_run", (((1,), (1,)), "float16", "less_equal")),
            # DeepLabV3 shapes
            # support int64 input to less_equal
            # ("004_less_equal", "less_equal_run",(((263169,),(263169,)),"int64","less_equal"), ((128, 128),)),
            ("005_less_equal", "less_equal_run", (((64, 128, 768), (1,)), "float16", "less_equal")),
            ("006_less_equal", "less_equal_run", (((128, 128, 64), (1,)), "float16", "less_equal")),
            ("007_less_equal", "less_equal_run", (((2, 1), (3, 3, 1, 3)), "float16", "less_equal")),
        ]

        self.testarg_rpc_cloud = [
            ("003_less_equal", "less_equal_run", (((1,), (1,)), "int32", "less_equal")),
            ("002_less_equal", "less_equal_run", (((1,), (1,)), "float32", "less_equal")),
            ("003_less_equal", "less_equal_run", (((128, 128, 64), (128, 128, 64)), "float32", "less_equal")),
            ("004_less_equal", "less_equal_run", (((64, 128, 768), (1,)), "float32", "less_equal")),
            ("005_less_equal", "less_equal_run", (((128, 128, 64), (1,)), "float32", "less_equal")),
            ("less_equal_001", "less_equal_run", (((64, 128, 768), (64, 128, 768)), "float32", "less_equal")),
            ("less_equal_002", "less_equal_run", (((128, 128, 64), (128, 128, 64)), "float32", "less_equal")),

        ]

        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
