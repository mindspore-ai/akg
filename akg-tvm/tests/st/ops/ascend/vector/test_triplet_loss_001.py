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
from tests.common.test_run.ascend.triplet_loss_run import triplet_loss_run
from tests.common.test_run.ascend.triplet_loss_ad_run import triplet_loss_grad_run, triplet_loss_ad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_reduce_max"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("triplet_loss_0", triplet_loss_run, ((2, 3), "float16", 12.0, "triplet_loss_fp16")),
            ("triplet_loss_0", triplet_loss_run, ((5, 7, 3), "float16", 9.0, "triplet_loss_fp16")),
            ("triplet_loss_0", triplet_loss_grad_run, ((2, 3), "float16", 12.0, "triplet_loss_fp16")),
            ("triplet_loss_0", triplet_loss_grad_run, ((4, 5), "float16", 6.0, "triplet_loss_fp16")),
            ("triplet_loss_0", triplet_loss_ad_run, ((2, 3), "float16", 12.0, "triplet_loss_fp16")),
            ("triplet_loss_0", triplet_loss_ad_run, ((4, 5), "float16", 6.0, "triplet_loss_fp16")),
        ]

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
