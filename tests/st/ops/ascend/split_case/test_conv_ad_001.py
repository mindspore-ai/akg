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
from tests.common.test_run.conv_ad_v2_run import conv_ad_v2_run as conv_ad_01_run
from tests.common.test_run.conv_filter_ad_run import conv_filter_ad_run
from tests.common.test_run.conv_input_ad_run import conv_input_ad_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_conv_ad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs,
            # testflag, opfuncname,  case_num, fmap_shape              , filter_shape          , pad_                              , stride_   , dilation_ , use_bias, bypass_l1, dump_data, Tile
            # (testflag, opfuncname, (case_num, (in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w), bias    , bypass_l1, dump_data, [cutH, cutCo, cutM, cutK, cutN]))
            # ("conv_ad_01_001",  conv_ad_01_run, (1, (32, 16, 34, 34), (64, 16, 3, 3), (0, 0, 0, 0), (1, 1), (1, 1), False, True, False, [128, 128, 64, 128, 64])),
            # ("conv_ad_01_002",  conv_ad_01_run, (2, (32, 16, 34, 34), (64, 16, 3, 3), (0, 0, 0, 0), (1, 1), (1, 1), False, True, False, [128, 128, 64, 128, 64])),
            # testflag, opfuncname,  case_num, fmap_shape              , filter_shape          , pad_                              , stride_   , dilation_
            # (testflag, opfuncname, (case_num, (in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w)))
            ("conv_ad_01_003", conv_ad_01_run, (3, (32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_ad_01_004", conv_ad_01_run, (4, (32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            # testflag, opfuncname, fmap_shape               , filter_shape          , pad_                              , stride_   , dilation_
            # (testflag, opfuncname, ((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w)))
            # ("conv_ad_01_005",  conv_input_ad_reuse_forward_run, ((32, 16, 34, 34), (64, 16, 3, 3), (0, 0, 0, 0), (1, 1), (1, 1), [128, 128, 64, 128, 64])),
            # ("conv_ad_01_006",  conv_input_ad_reuse_forward_run, ((32, 16, 33, 33), (64, 16, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), [128, 128, 64, 128, 64])),
            # testflag, opfuncname, fmap_shape               , filter_shape          , pad_                              , stride_   , dilation_ 
            # (testflag, opfuncname, ((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w)))
            ("conv_ad_01_007", conv_input_ad_run, ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_ad_01_008", conv_input_ad_run, ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            # testflag, opfuncname, fmap_shape               , filter_shape          , pad_                              , stride_   , dilation_
            # (testflag, opfuncname, ((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w)))
            ("conv_ad_01_009", conv_filter_ad_run,
             ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_ad_01_010", conv_filter_ad_run, ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
        ]

        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs,
            # testflag, opfuncname,  case_num, fmap_shape              , filter_shape          , pad_                              , stride_   , dilation_ , use_bias, bypass_l1, dump_data, Tile
            # (testflag, opfuncname, (case_num, (in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w), bias    , bypass_l1, dump_data, [cutH, cutCo, cutM, cutK, cutN]))
            # ("conv_ad_01_001",  conv_ad_01_run, (1, (32, 16, 34, 34), (64, 16, 3, 3), (0, 0, 0, 0), (1, 1), (1, 1), False, True, False, [128, 128, 64, 128, 64])),
            # ("conv_ad_01_002",  conv_ad_01_run, (2, (32, 16, 34, 34), (64, 16, 3, 3), (0, 0, 0, 0), (1, 1), (1, 1), False, True, False, [128, 128, 64, 128, 64])),
            # ("conv_ad_01_001b", conv_ad_01_run, (1, (32, 16, 33, 33), (64, 16, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), False, True, False, [128, 128, 64, 128, 64])),
            # ("conv_ad_01_002b", conv_ad_01_run, (2, (32, 16, 33, 33), (64, 16, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), False, True, False, [128, 128, 64, 128, 64])),
            # testflag, opfuncname,  case_num, fmap_shape              , filter_shape          , pad_                              , stride_   , dilation_
            # (testflag, opfuncname, (case_num, (in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w)))
            ("conv_filter_ad_run_000", conv_filter_ad_run,
             ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_filter_ad_run_001", conv_filter_ad_run,
             ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_002", conv_filter_ad_run,
             ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_filter_ad_run_003", conv_filter_ad_run,
             ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_filter_ad_run_004", conv_filter_ad_run,
             ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_005", conv_filter_ad_run,
             ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_006", conv_filter_ad_run,
             ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_007", conv_filter_ad_run,
             ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_filter_ad_run_008", conv_filter_ad_run,
             ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_filter_ad_run_009", conv_filter_ad_run,
             ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_010", conv_filter_ad_run,
             ((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))),
            ("conv_filter_ad_run_011", conv_filter_ad_run,
             ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_012", conv_filter_ad_run,
             ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_filter_ad_run_013", conv_filter_ad_run,
             ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_014", conv_filter_ad_run,
             ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_filter_ad_run_015", conv_filter_ad_run,
             ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_016", conv_filter_ad_run,
             ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_filter_ad_run_017", conv_filter_ad_run,
             ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_filter_ad_run_018", conv_filter_ad_run,
             ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_filter_ad_run_019", conv_filter_ad_run,
             ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),

            ("conv_input_ad_run_000", conv_input_ad_run,
             ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_input_ad_run_001", conv_input_ad_run,
             ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_002", conv_input_ad_run,
             ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_input_ad_run_003", conv_input_ad_run,
             ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_input_ad_run_004", conv_input_ad_run,
             ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_005", conv_input_ad_run,
             ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_006", conv_input_ad_run,
             ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_007", conv_input_ad_run,
             ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_input_ad_run_008", conv_input_ad_run,
             ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_input_ad_run_009", conv_input_ad_run,
             ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_010", conv_input_ad_run,
             ((1, 3, 224, 224), (64, 3, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1))),
            ("conv_input_ad_run_011", conv_input_ad_run,
             ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_012", conv_input_ad_run,
             ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_input_ad_run_013", conv_input_ad_run,
             ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_014", conv_input_ad_run,
             ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_input_ad_run_015", conv_input_ad_run,
             ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_016", conv_input_ad_run,
             ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1))),
            ("conv_input_ad_run_017", conv_input_ad_run,
             ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1))),
            ("conv_input_ad_run_018", conv_input_ad_run,
             ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
            ("conv_input_ad_run_019", conv_input_ad_run,
             ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1))),
        ]

        self.testarg_level2 = [
            # Fail in MakeAPI: ("conv_ad_1_3_8_8", conv_topi_input_ad, ((1, 3, 8, 8), (1, 3, 4, 4), (2, 2), (0, 0), "float16"), [(16, 65535), (16, 65535), (16, 0), (16, 0)]),
            ## Additional test cases from issues #1106
            ("test_resnet50_conv_input_ad_022", conv_input_ad_run,
             ((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))),
            ("test_resnet50_conv_input_ad_023", conv_input_ad_run,
             ((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))),
            ("test_resnet50_conv_input_ad_024", conv_input_ad_run,
             ((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))),
            ("test_resnet50_conv_input_ad_022", conv_filter_ad_run,
             ((32, 128, 56, 56), (128, 128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))),
            ("test_resnet50_conv_input_ad_023", conv_filter_ad_run,
             ((32, 256, 28, 28), (256, 256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))),
            ("test_resnet50_conv_input_ad_024", conv_filter_ad_run,
             ((32, 512, 14, 14), (512, 512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1))),
            # Add new dims to support hard case in conv_backprop_input and conv_input_ad
            ("conv_input_ad_run_010", conv_input_ad_run,
             ((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1))),
            # alex_net from issue 1142
            ("test_alexnet_conv_filter_ad_000", conv_filter_ad_run,
             ([32, 3, 227, 227], [96, 3, 11, 11], (0, 0, 0, 0), (4, 4), (1, 1))),
            ("test_alexnet_conv_filter_ad_000", conv_filter_ad_run,
             ([32, 3, 227, 227], [96, 3, 11, 11], (0, 0, 0, 0), (4, 4), (1, 1))),

        ]

        return

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

    def test_run_level2(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level2)

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
    a.test(3, 0)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test1():
    a = TestCase()
    a.setup()
    a.test(3, 1)
    a.teardown()


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test2():
    a = TestCase()
    a.setup()
    a.test(3, 2)
    a.teardown()
