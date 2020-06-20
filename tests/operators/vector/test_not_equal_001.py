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
from test_run.not_equal_run import not_equal_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_auto_not_equal_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_not_equal", not_equal_run, (((128,), (128,)), "float16", "not_equal")),
            ("002_not_equal", not_equal_run, (((128, 128), (128, 128)), "float16", "not_equal")),
            ("003_not_equal", not_equal_run, (((1,), (1,)), "float16", "not_equal")),
            # DeepLabV3 shapes
            ("004_not_equal", not_equal_run, (((21,), (21,)), "float16", "not_equal")),
            # support int32 input to not_equal
            # ("005_not_equal", not_equal_run, (((263169,), (263169,)), "int32", "not_equal"), ((128,128),(128,128))),
            # ("006_not_equal", not_equal_run, (((1052676,), (1052676,)), "int32", "not_equal"), ((128,128),(128,128))),

        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
