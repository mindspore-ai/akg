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

"""testcase for conv_backprop_filter op"""

import os
import pytest
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.conv_backprop_filter_run import conv_backprop_filter_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_conv_backprop_filter_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("conv_backprop_filter_run_001", conv_backprop_filter_run, ((32, 6, 14, 14), (16, 6, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_001", conv_backprop_filter_run, ((2, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))),

            # testflag, opfuncname, testRunArgs,
            # testflag, opfuncname, fmap_shape               , filter_shape          , pad_                              , stride_   , dilation_ , Tile
            #(testflag, opfuncname, ((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w), [cutCi, cutKH, cutKW, cutCo, cutBatch, cutH, cutW, cutM, cutK, cutN]))

            # resnet50 conv backprop filter prop testcase
            ("conv_backprop_filter_run_000", conv_backprop_filter_run, ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_backprop_filter_run_001", conv_backprop_filter_run, ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_002", conv_backprop_filter_run, ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_backprop_filter_run_003", conv_backprop_filter_run, ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_004", conv_backprop_filter_run, ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_005", conv_backprop_filter_run, ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_006", conv_backprop_filter_run, ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_007", conv_backprop_filter_run, ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_008", conv_backprop_filter_run, ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_backprop_filter_run_009", conv_backprop_filter_run, ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_010", conv_backprop_filter_run, ((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))),
            ("conv_backprop_filter_run_011", conv_backprop_filter_run, ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_012", conv_backprop_filter_run, ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_backprop_filter_run_013", conv_backprop_filter_run, ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_014", conv_backprop_filter_run, ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_015", conv_backprop_filter_run, ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_016", conv_backprop_filter_run, ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_017", conv_backprop_filter_run, ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_backprop_filter_run_018", conv_backprop_filter_run, ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_backprop_filter_run_019", conv_backprop_filter_run, ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
        ]
        return

    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg, is_conv=True)

    def test(self, split_nums, split_idx):
        self.common_run(get_splitted_cases(self.testarg, split_nums, split_idx), is_conv=True)

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
    a.test(4, 0)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test1():
    a = TestCase()
    a.setup()
    a.test(4, 1)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test2():
    a = TestCase()
    a.setup()
    a.test(4, 2)
    a.teardown()


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test3():
    a = TestCase()
    a.setup()
    a.test(4, 3)
    a.teardown()
