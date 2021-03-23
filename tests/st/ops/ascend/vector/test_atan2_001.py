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

"""atan2 test case"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.atan2_run import atan2_run


class TestAtan2(TestBase):

    def setup(self):
        """setup case parameters for test"""
        case_name = "test_akg_atan2_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("=================%s Setup case=================", self.casename)
        self.testarg_mini = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("atan2_f16_01", atan2_run, ((8, 16), "float16", (8, 16), "float16")),
            ("atan2_f32_02", atan2_run, ((8, 16), "float32", (8, 16), "float32")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("atan2_f16_03", atan2_run, ((32, 256, 16), "float16", (32, 256, 16), "float16")),
            ("atan2_f32_04", atan2_run, ((32, 256, 16), "float32", (32, 256, 16), "float32")),
        ]

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_mini_run(self):
        """run case for mini"""
        self.common_run(self.testarg_mini)

    def test_cloud_run(self):
        """run case for cloud"""
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """clean environment"""
        self._log.info("=============%s Teardown===========", self.casename)
