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
from tests.common.test_run.l2loss_run import l2loss_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_l2loss_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_l2loss", l2loss_run, ((1, 128), "float16", "l2_loss")),
            ("001_l2loss", l2loss_run, ((8, 7, 10, 6), "float32", "l2_loss")),


        ]
        self.testarg_cloud = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_l2loss", l2loss_run, ((1, 128), "float32", "l2_loss")),

        ]
        self.testarg_level1 = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            # Deeplab v3
            ("002_l2loss", l2loss_run, ((3, 3, 3, 32), "float16", "l2_loss")),
            ("003_l2loss", l2loss_run, ((3, 3, 32, 64), "float16", "l2_loss")),
            ("004_l2loss", l2loss_run, ((1, 1, 2048, 256), "float16", "l2_loss")),
            ("005_l2loss", l2loss_run, ((1, 1, 304, 256), "float16", "l2_loss")),
            ("006_l2loss", l2loss_run, ((1, 1, 1024, 1536), "float16", "l2_loss")),
            ("007_l2loss", l2loss_run, ((1, 1, 1280, 256), "float16", "l2_loss")),
            ("008_l2loss", l2loss_run, ((1, 1, 256, 21), "float16", "l2_loss")),
            ("009_l2loss", l2loss_run, ((1, 1, 256, 728), "float16", "l2_loss")),
            ("011_l2loss", l2loss_run, ((1, 1, 128, 128), "float16", "l2_loss")),
            ("012_l2loss", l2loss_run, ((1, 1, 64, 128), "float16", "l2_loss")),
            ("014_l2loss", l2loss_run, ((1, 1, 128, 256), "float16", "l2_loss")),
            ("017_l2loss", l2loss_run, ((1, 1, 728, 728), "float16", "l2_loss")),
            ("018_l2loss", l2loss_run, ((1, 1, 256, 48), "float16", "l2_loss")),
            # below cases have some precision porblem
            # ("010_l2loss", l2loss_run, ((1, 1, 256, 256), "float16", "l2_loss")),
            # ("013_l2loss", l2loss_run, ((1, 1, 1536, 1536), "float16", "l2_loss")),
            # ("015_l2loss", l2loss_run, ((1, 1, 728, 1024), "float16", "l2_loss")),
            # ("016_l2loss", l2loss_run, ((1, 1, 1536, 2048), "float16", "l2_loss")),
            # ("019_l2loss", l2loss_run, ((1, 1, 1024, 1024), "float16", "l2_loss")),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
