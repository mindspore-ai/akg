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
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.ascend.maxpool_ad_run import maxpool_ad_run


############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_maxpool_ad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path, 0)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # test case for first max
            # testflag,opfuncname,testRunArgs:shape,kernel,stride,pad,dtype,optimized,polyhedral,first_max
            ("mansch_ad_fp16_01_firstmax", maxpool_ad_run,
             ((1, 1, 32, 32, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float16", True, False, True)),
            ("mansch_ad_fp16_01_firstmax", maxpool_ad_run,
             ((2, 2, 112, 32, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float16", True, False, True)),
            ("mansch_ad_fp16_01_firstmax", maxpool_ad_run,
             ((1, 1, 10, 10, 16), (2, 2), (2, 2), "VALID", "float16", True, False, True)),
            ("mansch_ad_fp16_01_firstmax", maxpool_ad_run,
             ((1, 1, 28, 28, 16), (2, 2), (2, 2), "VALID", "float16", True, False, True)),
            ("mansch_ad_fp16_01_firstmax", maxpool_ad_run,
             ((1, 1, 28, 28, 16), (2, 2), (2, 2), (1, 1, 1, 1), "float16", True, False, True)),

            # test case for all max
            # ("mansch_ad_fp16_01", maxpool_ad_run, ((1, 1, 16, 16, 16), (3, 3), (2, 2), (1, 1, 1, 1), "float16", True, False, False)),

            # test cases for all max using AD (no custom diff)
            # poly
            # ("mansch_ad_fp16_01", maxpool_ad_run, ((1, 1, 17, 17, 16), (3, 3), (2, 2), (0, 0, 0, 0), "float16", False, True, False)),
            # manual schedule
            ("mansch_ad_fp16_01", maxpool_ad_run,
             ((1, 1, 17, 17, 16), (3, 3), (2, 2), (0, 0, 0, 0), "float16", False, False, False)),
            # ("mansch_ad_fp16_01", maxpool_ad_run, ((2, 4, 17, 17, 16), (3, 3), (2, 2), (0, 0, 0, 0), "float16", False, False, False)),
        ]

        self.testarg_level1 = [
            # Resnet50
            # testflag,opfuncname,testRunArgs:shape,kernel,stride,pad,dtype,optimized,polyhedral,first_max
            # first max cases
            ("mansch_ad_fp16_00_firstmax", maxpool_ad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), "SAME", "float16", True, False, True)),
            ("mansch_ad_fp16_01_firstmax", maxpool_ad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float16", True, False, True)),
            ("mansch_ad_fp16_02_firstmax", maxpool_ad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 0, 1, 0), "float16", True, False, True)),

            # all max cases
            # ("mansch_ad_fp16_01_allmax", maxpool_ad_run, ((1, 1, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), "float16", True, False, False)),
            # ("mansch_ad_fp16_02_allmax", maxpool_ad_run, ((1, 4, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), "float16", True, False, False)),
            # ("mansch_ad_fp16_03_allmax", maxpool_ad_run, ((32, 1, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), "float16", True, False, False)),
            # ("mansch_ad_fp16_04_allmax", maxpool_ad_run, ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), "float16", True, False, False)),
        ]

        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs:shape,kernel,stride,pad,dtype,optimized,polyhedral,first_max
            ("mansch_ad_fp16_00_firstmax", maxpool_ad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), "SAME", "float16", True, False, True)),
            ("mansch_ad_fp16_01_firstmax", maxpool_ad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float16", True, False, True)),
            ("mansch_ad_fp16_02_firstmax", maxpool_ad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 0, 1, 0), "float16", True, False, True)),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def test_level1(self, split_nums, split_idx):
        self.common_run(get_splitted_cases(self.testarg_level1, split_nums, split_idx))

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test0_level1():
    a = TestCase()
    a.setup()
    a.test_level1(3, 0)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test1_level1():
    a = TestCase()
    a.setup()
    a.test_level1(3, 1)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test2_level1():
    a = TestCase()
    a.setup()
    a.test_level1(3, 2)
    a.teardown()


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
