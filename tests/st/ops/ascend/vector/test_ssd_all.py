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
from tests.common.base import TestBase
from tests.common.test_run.ascend.focal_loss_run import focal_loss_run
from tests.common.test_run.ascend.focalloss_ad_run import focalloss_grad_run
from tests.common.test_run.ascend.smooth_l1_loss_run import smooth_l1_loss_run
from tests.common.test_run.ascend.smooth_l1_loss_grad_run import smooth_l1_loss_grad_run
from tests.common.test_run import reduce_sum_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestFocalLoss(TestBase):
    def setup(self):
        case_name = "test_akg_ssd_all"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_cloud = [
            ("test_ssd_001_one_hot_001", "one_hot_run", ((32, 8732), 6, "float16", 1, 0, -1)),
            ("test_ssd_002_one_hot_002", "one_hot_run", ((32, 8732), 6, "float32", 1, 0, -1)),
            ("test_ssd_003_sum_001", reduce_sum_run, ((32, 8732), (-1,), False, "float16")),
            ("test_ssd_004_mean_001", "mean_run", ((32,), "float16", (0,), False, "reduce_mean")),
            #  ("test_ssd_004_mean_001", "mean_run", ((32,), "float16", (0,), True, "reduce_mean")),
            ("test_ssd_005_focal_loss_001", focal_loss_run, ((32, 8732, 6), "float16", "float16", 2.0, "focal_loss")),
            ("test_ssd_006_focalloss_grad_001", focalloss_grad_run, ((32, 8732, 6), "float16", "float16", 2)),
            ("test_ssd_007_smooth_l1_loss_001", smooth_l1_loss_run, ((32, 8732, 4), "float16", (32, 8732, 4), "float16", (32, 8732), "int32", 0, 1.0, "smooth_l1_loss")),
            ("test_ssd_008_smooth_l1_loss_grad_001", smooth_l1_loss_grad_run, ((32, 8732, 4), "float16")),
        ]
        return

    def test_cloud_run(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    c = TestFocalLoss()
    c.setup()
    c.test_ci_run()
    c.teardown()
