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
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.ascend.depthwise_run import depthwise_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_depthwise_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs
            # CO = k_ch * CI
            # group = CI // block_size

            # (testflag, opfuncname,(N, H, W, CI, k_ch, KH, KW, pad_h, pad_w, stride_h, stride_w, cutH, cutCo, cutM, cutK, cutN)

            ("depthwise_run_001", depthwise_run, (16, 112, 112, 32, 1, 3, 3, 1, 1, 1, 1)),

            ("depthwise_run_002", depthwise_run, (16, 112, 112, 64, 1, 3, 3, 1, 1, 2, 2)),

            ("depthwise_run_003", depthwise_run, (1, 56, 56, 128, 1, 3, 3, 1, 1, 1, 1)),

            ("depthwise_run_004", depthwise_run, (1, 56, 56, 128, 1, 3, 3, 1, 1, 2, 2)),

            ("depthwise_run_005", depthwise_run, (1, 28, 28, 256, 1, 3, 3, 1, 1, 1, 1)),

            ("depthwise_run_006", depthwise_run, (1, 28, 28, 256, 1, 3, 3, 1, 1, 2, 2)),

            ("depthwise_run_007", depthwise_run, (1, 14, 14, 512, 1, 3, 3, 1, 1, 1, 1)),

            ("depthwise_run_008", depthwise_run, (1, 14, 14, 512, 1, 3, 3, 1, 1, 2, 2)),
            # fail in general value of " cutH, cutCo, cutM, cutK, cutN"
            ("depthwise_run_009", depthwise_run, (1, 7, 7, 1024, 1, 3, 3, 1, 1, 1, 1)),
        ]
        self.testarg_1 = [
            # # mobilenet V2

            ("depthwise_run_101", depthwise_run, (1, 112, 112, 32, 1, 3, 3, 1, 1, 1, 1)),

            ("depthwise_run_102", depthwise_run, (1, 112, 112, 96, 1, 3, 3, 1, 1, 2, 2)),

            # fail ("depthwise_run_103", depthwise_run,(1,  56,  56, 144, 1, 3, 3, 1, 1, 1, 1, 58, 16, 176, 96, 16)),

            # fail ("depthwise_run_104", depthwise_run,(1,  56,  56, 144, 1, 3, 3, 1, 1, 2, 2, 58, 16, 14*16, 9*16, 16)),

            # fail ("depthwise_run_105", depthwise_run,(1,  28,  28, 192, 1, 3, 3, 1, 1, 1, 1, 30, 16, 16*7, 9*16, 16)),

            # fail ("depthwise_run_106", depthwise_run,(1,  28,  28, 192, 1, 3, 3, 1, 1, 2, 2, 30, 16, 512, 3*16, 16)),

            # fail ("depthwise_run_107", depthwise_run,(1,  14,  14, 384, 1, 3, 3, 1, 1, 1, 1, 16, 16, 160, 80, 16)),

            # fail ("depthwise_run_108", depthwise_run,(1,  14,  14, 576, 1, 3, 3, 1, 1, 1, 1, 16, 16, 208, 9*16, 16)),

            # fail ("depthwise_run_109", depthwise_run,(1,  14,  14, 576, 1, 3, 3, 1, 1, 2, 2, 16, 16, 64, 9*16, 16)),
            # fail in general value of " cutH, cutCo, cutM, cutK, cutN"
            ("depthwise_run_110", depthwise_run, (1, 7, 7, 960, 1, 3, 3, 1, 1, 1, 1)),

        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def test_run_1(self):
        self.common_run(self.testarg_1)

    def test(self, split_nums, split_idx):
        self.common_run(get_splitted_cases(self.testarg, split_nums, split_idx))

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test0():
    a = TestCase()
    a.setup()
    a.test(2, 0)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test1():
    a = TestCase()
    a.setup()
    a.test(2, 1)
    a.teardown()
