# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.common.test_run.prod_force_se_a_grad_run import prod_force_se_a_grad_run


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
            ("prod_force_se_a_grad_01", prod_force_se_a_grad_run, ([[8, 192, 3], [8, 192, 144 * 4, 3], [8, 192, 144]], ["float32", "float32", "int32"]), ["level0"]),
            ("prod_force_se_a_grad_02", prod_force_se_a_grad_run, ([[1, 192, 3], [1, 192, 138 * 4, 3], [1, 192, 138]], ["float32", "float32", "int32"]), ["level0"])
        ]

        if not super(TestCase, self).setup():
            return False
        self._log.info("TestCase:{0} Setup case".format(self.casename))
        return True

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_level0(self):
        return self.run_test_arg_func(self.test_args, "level0")

    def test_level1(self):
        return self.run_test_arg_func(self.test_args, "level1")

    def test_run_cloud(self):
        return self.run_test_arg_func(self.test_args, "aic_cloud")

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return


def print_args():
    cls = TestCase()
    cls.print_args()
