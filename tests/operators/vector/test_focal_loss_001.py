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
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.focal_loss_run import focal_loss_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestFocalLoss(TestBase):

    def setup(self):
        case_name = "test_akg_focal_loss_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_ci = [
            # testflag,opfuncname,testRunArgs, dimArgs
            #  ("focal_loss_001", focal_loss_run, ((8,4718,6),  "float16", "float16", 2.0, "focalloss_1")),
            #  ("focal_loss_002", focal_loss_run, ((8,4718,6),  "float32", "float32", 2.0, "focalloss_2")),
            #  ("focal_loss_003", focal_loss_run, ((8,4718,12), "float16", "float16", 2.0, "focalloss_3")),
            ("focal_loss_004", focal_loss_run, ((8, 4718, 12), "float32", "float32", 2.0, "focalloss_4")),
            ("focal_loss_005", focal_loss_run, ((8, 4718, 12), "float32", "int32", 2.0, "focalloss_5")),
            ("focal_loss_005", focal_loss_run, ((8, 8732, 6), "float16", "float16", 2.0, "focalloss_c_5")),
            ("focal_loss_006", focal_loss_run, ((8, 8732, 6), "float32", "float32", 2.0, "focalloss_c_6")),
            # following case should fail!
            #  ("focal_loss_006", focal_loss_run, ((8,4718,12), "float32", "float16", 2.0, "focalloss_6")),
            #  ("focal_loss_007", focal_loss_run, ((8,4718,12), "float32", "int16",   2.0, "focalloss_7")),
            #  ("focal_loss_008", focal_loss_run, ((8,4718,12), "float16", "float32", 2.0, "focalloss_8")),
        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("focal_loss_001", focal_loss_run, ((8, 4718, 6), "float16", "float16", 2.0, "focalloss_c_1")),
            ("focal_loss_002", focal_loss_run, ((8, 4718, 6), "float32", "float32", 2.0, "focalloss_c_2")),
            ("focal_loss_003", focal_loss_run, ((8, 4718, 12), "float16", "float16", 2.0, "focalloss_c_3")),
            ("focal_loss_004", focal_loss_run, ((8, 4718, 12), "float32", "float32", 2.0, "focalloss_c_4")),
            ("focal_loss_005", focal_loss_run, ((8, 8732, 6), "float16", "float16", 2.0, "focalloss_c_5")),
            ("focal_loss_006", focal_loss_run, ((8, 8732, 6), "float32", "float32", 2.0, "focalloss_c_6")),
            ("focal_loss_012", focal_loss_run, ((8, 8732, 6), "float32", "float32", 2.5, "focalloss_c_12")),
            ("focal_loss_013", focal_loss_run, ((8, 8732, 6), "float32", "float32", 3.0, "focalloss_c_13")),
            ("focal_loss_007", focal_loss_run, ((8, 4718, 12), "float32", "int32", 2.0, "focalloss_c_7")),
            ("focal_loss_008", focal_loss_run, ((8, 8732, 6), "float16", "int32", 2.0, "focalloss_c_8")),
            ("focal_loss_009", focal_loss_run, ((8, 8732, 6), "float32", "int32", 2.0, "focalloss_c_9")),
            ("focal_loss_010", focal_loss_run, ((8, 4718, 12), "float16", "int32", 2.0, "focalloss_c_10")),
            ("focal_loss_008", focal_loss_run, ((32, 8732, 6), "float16", "float16", 2.0, "focalloss_c_11")),
            ("focal_loss_008", focal_loss_run, ((32, 8732, 6), "float32", "float32", 2.0, "focalloss_c_12")),
            ("focal_loss_008", focal_loss_run, ((32, 8732, 6), "float32", "float32", 2.5, "focalloss_c_13")),
            # following case should fail!
            #  ("focal_loss_101", focal_loss_run, ((8,4718,12), "float32", "int16",   2.0, "focalloss_c_fail")),
            #  ("focal_loss_102", focal_loss_run, ((8,8732,6),  "float16", "float32", 2.0, "focalloss_c_fail")),
            #  ("focal_loss_103", focal_loss_run, ((8,8732,6),  "float32", "float16", 2.0, "focalloss_c_fail")),
        ]
        self.testarg_aic_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("focal_loss_001", focal_loss_run, ((8, 59, 6), "float16", "float16", 2.0, "focalloss_aic_1")),
            ("focal_loss_002", focal_loss_run, ((8, 59, 6), "float32", "float32", 2.0, "focalloss_aic_2")),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_ci_run(self):
        self.common_run(self.testarg_ci)

    @pytest.mark.aic_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_aic_cloud_run(self):
        self.common_run(self.testarg_aic_cloud)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
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
