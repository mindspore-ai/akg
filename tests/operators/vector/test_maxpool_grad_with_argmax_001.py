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

from base import TestBase
import pytest
from test_run.maxpool_grad_with_argmax_run import maxpool_grad_with_argmax_run
import datetime

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_maxpool_ad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # test case for first max
            # testflag,opfuncname,testRunArgs:
            # shape, kernel, stride, pad, dtype, polyhedral
            ("mansch_ad_fp16_00_firstmax", maxpool_grad_with_argmax_run, \
            ((32, 16, 14, 14, 16), (3, 3), (2, 2), "SAME", "float32", False)),

            # alexnet
            ("mansch_ad_fp16_00_alexnet", maxpool_grad_with_argmax_run, \
            ((32, 16, 13, 13, 16), (3, 3), (2, 2), "VALID", "float16", False)),
            ("mansch_ad_fp16_01_alexnet", maxpool_grad_with_argmax_run, \
            ((32, 16, 27, 27, 16), (3, 3), (2, 2), "VALID", "float16", False)),
            ("mansch_ad_fp16_02_alexnet", maxpool_grad_with_argmax_run, \
            ((32, 6, 55, 55, 16), (3, 3), (2, 2), "VALID", "float16", False)),

            # lenet
            ("mansch_ad_fp16_00_lenet", maxpool_grad_with_argmax_run, \
            ((32, 1, 10, 10, 16), (2, 2), (2, 2), "VALID", "float16", False)),
            ("mansch_ad_fp16_01_lenet", maxpool_grad_with_argmax_run, \
            ((32, 1, 28, 28, 16), (2, 2), (2, 2), "VALID", "float16", False)),
        ]

        self.testarg_level1 = [
            # testflag,opfuncname,testRunArgs:
            # shape, kernel, stride, pad, dtype, polyhedral
            # Resnet50
            ("mansch_ad_fp16_00_resnet50", maxpool_grad_with_argmax_run, \
            ((32, 4, 112, 112, 16), (3, 3), (2, 2), "SAME", "float16", False)),
            ("mansch_ad_fp16_01_resnet50", maxpool_grad_with_argmax_run, \
            ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float16", False)),
            ("mansch_ad_fp16_02_resnet50", maxpool_grad_with_argmax_run, \
            ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 0, 1, 0), "float16", False)),
        ]

        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs:
            # shape, kernel, stride, pad, dtype, polyhedral
            # Resnet50
            ("mansch_ad_fp32_00_resnet50", maxpool_grad_with_argmax_run, \
            ((32, 4, 112, 112, 16), (3, 3), (2, 2), "SAME", "float32", False)),
            ("mansch_ad_fp32_01_resnet50", maxpool_grad_with_argmax_run, \
            ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), "float32", False)),
            ("mansch_ad_fp32_02_resnet50", maxpool_grad_with_argmax_run, \
            ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 0, 1, 0), "float32", False)),


            # alexnet
            ("mansch_ad_fp32_00_alexnet", maxpool_grad_with_argmax_run, \
            ((32, 16, 13, 13, 16), (3, 3), (2, 2), "VALID", "float32", False)),
            ("mansch_ad_fp32_01_alexnet", maxpool_grad_with_argmax_run, \
            ((32, 16, 27, 27, 16), (3, 3), (2, 2), "VALID", "float32", False)),
            ("mansch_ad_fp32_02_alexnet", maxpool_grad_with_argmax_run, \
            ((32, 6, 55, 55, 16), (3, 3), (2, 2), "VALID", "float32", False)),

            # lenet
            ("mansch_ad_fp32_00_lenet", maxpool_grad_with_argmax_run, \
            ((32, 1, 10, 10, 16), (2, 2), (2, 2), "VALID", "float32", False)),
            ("mansch_ad_fp32_01_lenet", maxpool_grad_with_argmax_run, \
            ((32, 1, 28, 28, 16), (2, 2), (2, 2), "VALID", "float32", False)),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level2
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.level2
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
