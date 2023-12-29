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
from tests.common.test_run.ascend.crossentropyloss_ad_run import crossentropyloss_ad_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_auto_diff_cross_entropy_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))

        self.testarg = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # shape, axis, dtype, kernal_name, attrs
            ("crossentropyloss_ad_run_1", crossentropyloss_ad_run, ((128, 128), "float16", "crossentropyloss_cce_f16")),
            ("crossentropyloss_ad_run_2", crossentropyloss_ad_run, ((3125, 512), "float16", "crossentropyloss_cce_f16")),
            ("crossentropyloss_ad_run_3", crossentropyloss_ad_run, ((1563, 512), "float16", "crossentropyloss_cce_f16")),
            ("crossentropyloss_ad_run_4", crossentropyloss_ad_run, ((15625, 512), "float16", "crossentropyloss_cce_f16")),
            ("crossentropyloss_ad_run_5", crossentropyloss_ad_run, ((31250, 512), "float16", "crossentropyloss_cce_f16")),
        ]

        self.testarg_cloud = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # shape, axis, dtype, kernal_name, attrs
            ("crossentropyloss_ad_run_1", crossentropyloss_ad_run, ((32, 64), "float16", "crossentropyloss_ad_f16"),),
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

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
