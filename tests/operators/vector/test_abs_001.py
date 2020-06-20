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

from base import TestBase
from test_run.abs_run import abs_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_abs_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.test_args = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("000_abs_input_1_1", abs_run, ((1, 1), "float16"), ["level0"]),
            ("001_abs_input_2_1", abs_run, ((2, 1), "float16"), ["level0"]),
            ("002_abs_input_2_2_2", abs_run, ((2, 2, 2), "float16"), ["level0"]),
            ("003_abs_input_1280_1280", abs_run, ((1280, 1280), "float16"), ["level0"]),
            ("004_abs_input_1280_30522", abs_run, ((1280, 30522), "float16"), ["level1"]),
            ("005_abs_input_8192_1024", abs_run, ((8192, 1024), "float16"), ["level1"]),
            ("006_abs_input_1280_1024", abs_run, ((1280, 1024), "float16"), ["level1"]),
            ("007_abs_input_64_128_1024", abs_run, ((64, 128, 1024), "float16"), ["level1"]),
        ]

        if not super(TestCase, self).setup():
            return False
        self._log.info("TestCase:{0} Setup case".format(self.casename))
        return True

    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_level0(self):
        return self.run_test_arg_func(self.test_args, "level0")

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_level1(self):
        return self.run_test_arg_func(self.test_args, "level1")

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        return self.run_test_arg_func(self.test_args, "aic_cloud")

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return


def print_args():
    cls = TestCase()
    cls.print_args()
