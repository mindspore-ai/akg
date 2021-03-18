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
from tests.common.test_run.equal_run import equal_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_auto_equal_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_equal", equal_run, (((128,), (128,)), "float16", "equal")),
            ("002_equal", equal_run, (((128, 128), (128, 128)), "float16", "equal")),
            ("003_equal", equal_run, (((1,), (1,)), "float16", "equal")),
            # DeepLabV3 shapes
            ("004_equal", equal_run, (((1052676,), (1052676,)), "float16", "equal")),
            # support int32 input to equal
            # ("005_equal", equal_run,(((263169,),(263169,)),"int32","equal"), ((128,128),(128, 128),)),
            # Bert shapes from TF(64)
            # ("006_equal", equal_run,(((160,),(160,)),"int32","equal"), ((128,128),(128, 128),)),
            # ("007_equal", equal_run,(((8,),(8)),"int32","equal"), ((128,128),(128, 128),)),

            # deeplabv3
            ("equal_1", equal_run, (((263169,), (1,)), "int32", "equal")),
            ("equal_2", equal_run, (((1,), (1,)), "int32", "equal")),
            ("equal_4", equal_run, (((1,), (1,)), "float32", "equal")),
            ("equal_6", equal_run, (((1052676,), (1,)), "float32", "equal")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
