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
from tests.common.test_run.group_conv_run import group_conv_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_group_conv_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs
            #CO = CI
            #group = CI // block_size
            #N, H, CI, K, PAD, STRIDE, cutH, cutCo, cutM, cutK, cutN

            # (testflag, opfuncname,(N, H, H, CI, CO, group, K, K, pad, pad, stride, stride, cutH, cutCo, cutM, cutK, cutN)
            # mobilenet V1
            ("group_conv_run_001", group_conv_run, (1, 7, 7, 1024, 1024, 64, 3, 3, 1, 1, 1, 1, 7, 16, 512, 3 * 16, 16)),

            # mobilenet V2
            ("group_conv_run_101", group_conv_run, (1, 7, 7, 960, 960, 60, 3, 3, 1, 1, 1, 1, 9, 16, 512, 3 * 16, 16)),

        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs
            # CO = CI
            # group = CI // block_size
            # N, H, CI, K, PAD, STRIDE, cutH, cutCo, cutM, cutK, cutN

            # (testflag, opfuncname,(N, H, H, CI, CO, group, K, K, pad, pad, stride, stride, cutH, cutCo, cutM, cutK, cutN)
            ("group_conv_run_001", group_conv_run, (1, 112, 112, 32, 32, 2, 3, 3, 1, 1, 1, 1, 15, 16, 256, 48, 16)),

            ("group_conv_run_002", group_conv_run, (1, 112, 112, 64, 64, 4, 3, 3, 1, 1, 2, 2, 15, 16, 512, 48, 16)),

            ("group_conv_run_003", group_conv_run, (1, 56, 56, 128, 128, 8, 3, 3, 1, 1, 1, 1, 15, 16, 512, 48, 16)),

            ("group_conv_run_004", group_conv_run, (1, 56, 56, 128, 128, 8, 3, 3, 1, 1, 2, 2, 15, 16, 512, 48, 16)),

            ("group_conv_run_005", group_conv_run, (1, 28, 28, 256, 256, 16, 3, 3, 1, 1, 1, 1, 7, 16, 512, 48, 16)),

            ("group_conv_run_006", group_conv_run, (1, 28, 28, 256, 256, 16, 3, 3, 1, 1, 2, 2, 7, 16, 512, 48, 16)),

            ("group_conv_run_007", group_conv_run, (1, 14, 14, 512, 512, 32, 3, 3, 1, 1, 1, 1, 7, 16, 512, 48, 16)),

            ("group_conv_run_008", group_conv_run, (1, 14, 14, 512, 512, 32, 3, 3, 1, 1, 2, 2, 7, 16, 512, 48, 16)),

            ("group_conv_run_009", group_conv_run, (1, 7, 7, 1024, 1024, 64, 3, 3, 1, 1, 1, 1, 7, 16, 512, 48, 16)),

            # mobilenet V2
            ("group_conv_run_101", group_conv_run, (1, 112, 112, 32, 32, 2, 3, 3, 1, 1, 1, 1, 114, 16, 256, 48, 16)),

            ("group_conv_run_102", group_conv_run, (1, 112, 112, 96, 96, 6, 3, 3, 1, 1, 2, 2, 114, 16, 196, 48, 16)),

            ("group_conv_run_103", group_conv_run, (1, 56, 56, 144, 144, 9, 3, 3, 1, 1, 1, 1, 58, 16, 196, 48, 16)),

            ("group_conv_run_104", group_conv_run, (1, 56, 56, 144, 144, 9, 3, 3, 1, 1, 2, 2, 58, 16, 112, 48, 16)),

            ("group_conv_run_105", group_conv_run, (1, 28, 28, 192, 192, 12, 3, 3, 1, 1, 1, 1, 30, 16, 112, 48, 16)),

            ("group_conv_run_106", group_conv_run, (1, 28, 28, 192, 192, 12, 3, 3, 1, 1, 2, 2, 30, 16, 512, 48, 16)),

            ("group_conv_run_107", group_conv_run, (1, 14, 14, 384, 384, 24, 3, 3, 1, 1, 1, 1, 16, 16, 196, 48, 16)),

            ("group_conv_run_108", group_conv_run, (1, 14, 14, 576, 576, 36, 3, 3, 1, 1, 1, 1, 16, 16, 196, 48, 16)),

            ("group_conv_run_109", group_conv_run, (1, 14, 14, 576, 576, 36, 3, 3, 1, 1, 2, 2, 16, 16, 49, 48, 16)),

            ("group_conv_run_110", group_conv_run, (1, 7, 7, 960, 960, 60, 3, 3, 1, 1, 1, 1, 9, 16, 512, 48, 16)),

        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    #a = TestCase("test_group_conv_conv_001", os.getcwd())
    a = TestCase()
    a.setup()
    a.test_run_level1()
    a.teardown()
