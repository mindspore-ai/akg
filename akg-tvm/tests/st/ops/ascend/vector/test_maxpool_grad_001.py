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

import os
import pytest
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.ascend.maxpool_grad_run import maxpool_grad_run


############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_maxpool_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs: shape, kernel, stride, pad, dtype, dimArgs
            # reid
            ("reid_mpgrad_fp16_01", maxpool_grad_run, ((2, 16, 40, 24, 16), (1, 1), (2, 2), (0, 0, 0, 0), "float16")),
            # others
            ("other_mpgrad_fp16_01", maxpool_grad_run, ((1, 1, 4, 4, 16), (2, 2), (1, 1), (0, 0, 0, 0), "float16")),

            ("lenet_MaxPoolV2Grad_32_1_10_10_16_f16_VALID_2_0_2", maxpool_grad_run, (
                (32, 1, 10, 10, 16), (2, 2), (2, 2), "VALID", "float16")),
            ("lenet_MaxPoolV2Grad_32_1_28_28_16_f16_VALID_2_0_2", maxpool_grad_run, (
                (32, 1, 28, 28, 16), (2, 2), (2, 2), "VALID", "float16")),

            ("other_mpgrad_fp16_01", maxpool_grad_run, ((1, 1, 4, 4, 16), (2, 2), (2, 2), (0, 0, 0, 0), "float16")),
            #  ("other_mpgrad_fp16_02", maxpool_grad_run, ((1, 1, 16, 16, 16), (4, 4), (4, 4), (0, 0, 0, 0), "float16")),
            #  ("other_mpgrad_fp16_03", maxpool_grad_run, ((1, 1, 32, 32, 16), (4, 4), (4, 4), (0, 0, 0, 0), "float16")),
        ]
        self.testarg_l1 = [
            # testflag,opfuncname,testRunArgs: shape, kernel, stride, pad, dtype, dimArgs
            # resnet 50
            ("resnet50_mpgrad_fp16_01", maxpool_grad_run, ((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', "float16")),
            ("resnet50_mpgrad_fp16_02", maxpool_grad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float16")),
            ("resnet50_mpgrad_fp16_03", maxpool_grad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 0, 1, 0), "float16")),
        ]

        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs: shape, kernel, stride, pad, dtype, dimArgs
            ("resnet50_mpgrad_fp32_01", maxpool_grad_run, ((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', "float32")),
            ("resnet50_mpgrad_fp32_02", maxpool_grad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float32")),
            ("resnet50_mpgrad_fp32_03", maxpool_grad_run,
             ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 0, 1, 0), "float32")),
        ]
        self.testarg_aic = [
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_level_1(self):
        self.common_run(self.testarg_l1)

    def test_level1(self, split_nums, split_idx):
        self.common_run(get_splitted_cases(self.testarg_l1, split_nums, split_idx))

    def test_run_aic(self):
        self.common_run(self.testarg_aic)

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
