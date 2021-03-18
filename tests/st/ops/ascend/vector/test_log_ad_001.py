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
from tests.common.base import TestBase
from tests.common.test_run.log_ad_run import log_ad_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_auto_diff_log"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))

        self.testarg = [
            # caseflag, opfuncname, testRunArgs, dimArg
            # shape, dtype, kernal_name, attrs
            ("log_ad_run_0", log_ad_run, ((32, 64), "float16", "log_ad_f16")),
        ]

        self.testarg_cloud = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # shape,  dtype, kernal_name, attrs
            ("log_ad_run_1", log_ad_run, ((32, 64), "float16", "log_ad_f16"),),
        ]

        self.testarg_level1 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("log_ad_run_2", log_ad_run, ((128, 512), "float16", "log_ad_f16"),),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
