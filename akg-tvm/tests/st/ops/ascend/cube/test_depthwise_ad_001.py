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
unsortedsegmentsum test cast
"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.depthwise_ad_run import depthwise_ad_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_autodiff_depthwise_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs
            # CO = CI
            # group = CI // block_size
            # N, H, CI, K, PAD, STRIDE, cutH, cutCo, cutM, cutK, cutN
            ("depthwise_ad_run_001", depthwise_ad_run, (16, 7, 7, 1024, 1, 3, 3, 1, 1, 1, 1, 7, 16, 512, 3 * 16, 16)),
        ]
        self.testarg1 = [
            ("depthwise_ad_run_101", depthwise_ad_run, (16, 7, 7, 960, 1, 3, 3, 1, 1, 1, 1, 9, 16, 512, 3 * 16, 16)),
        ]
        return

    @pytest.mark.skip
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.skip
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    #a = TestCase("test_depthwise_ad_001", os.getcwd())
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
