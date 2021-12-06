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

"""testcase for add_a_conv op"""

import os
import pytest
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.ascend.add_a_conv_run import add_a_conv_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_add_a_conv_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs,
            # testflag, opfuncname, fmap_shape               , filter_shape          , pad_                                       , stride_              , dilation_                , use_bias, bypass_l1 , dump_data, Tile
            # (testflag, opfuncname, ((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (pad_left, pad_right, pad_top, pad_bottom), (stride_h, stride_w), (dilation_h, dilation_w), bias    , bypass_l1 , dump_data, [cutH, cutCo, cutM, cutK, cutN]))

            # resnet50_wkl
            # k_h_d = (k_h - 1) * d_h + 1
            # tile_out_h = (tile_hh - k_h_d) // s_h + 1
            # tile_hh = (tile_out_h - 1) * s_h + k_h_d

            # ("simple_case_0", add_a_conv_run,((1  , 128 , 14  , 14  ), (128, 128 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), False,False, False , [  14  , 16*8, 64   , 96   , 128  ])),
            # ("simple_case_1", add_a_conv_run,((1  , 128 , 14  , 14  ), (128, 128 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), False,False, False , [  5  , 16*8, 64   , 96   , 128  ])),
            # ("add_a_conv_run001", add_a_conv_run,((1  , 1024 , 14  , 14  ), (2048, 1024 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True,True, False , [  5, 2048 , 64   , 96   , 128  ])),
            ("add_a_conv_run002", add_a_conv_run, (
            (1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, True, False,
            [7, 256, 208, 64, 112])),
            # ("add_a_conv_run003", add_a_conv_run,((1  , 1024 , 14  , 14  ), (512 , 1024 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True  ,True, False , [  5, 512  , 49   , 32   , 512  ])),
            # ("add_a_conv_run004", add_a_conv_run,((1  , 128  , 28  , 28  ), (128 , 128  , 3  , 3  ), (1 , 1 , 1 , 1 ), ( 1 , 1 ), (1, 1), False , False , False , [  16  , 128  , 400  , 32   , 128  ])),
            ("add_a_conv_run005", add_a_conv_run, (
            (1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False,
            [28, 512, 784, 16, 32])),
            ("add_a_conv_run006", add_a_conv_run, (
            (1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, True, False,
            [7, 512, 49, 32, 512])),
            ("add_a_conv_run007", add_a_conv_run, (
            (1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False,
            [14, 64 * 16, 112, 32, 240])),
            # ("add_a_conv_run008", add_a_conv_run,((1  , 256  , 14  , 14  ), (256 , 256  , 3  , 3  ), (1 , 1 , 1 , 1 ), ( 1 , 1 ), (1, 1), True  , True, False , [  14  , 256  , 196  , 64   , 256  ])),
            # ("add_a_conv_run009", add_a_conv_run,((1  , 256  , 56  , 56  ), (128 , 256  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True, False , False , [  7, 128  , 252  , 64   , 128  ])),
            ("add_a_conv_run010", add_a_conv_run, (
            (1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False, [8, 64, 280, 16, 64])),
            # ("add_a_conv_run011", add_a_conv_run,((1  , 3    , 224 , 224 ), (64  , 3    , 7  , 7  ), (3 , 3 , 3 , 3 ), ( 2 , 2 ), (1, 1), True  , False , False , [  9, 64   , 448  , 32   , 64   ])),
            ("add_a_conv_run012", add_a_conv_run, (
            (1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False,
            [7, 128, 448, 16, 64])),
            # ("add_a_conv_run013", add_a_conv_run,((1  , 512  , 28  , 28  ), (256 , 512  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True, False , False , [  3, 256  , 98   , 64   , 256  ])),
            ("add_a_conv_run014", add_a_conv_run, (
            (1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, True, False,
            [7, 2048, 49, 16, 512])),
            # ("add_a_conv_run015", add_a_conv_run,((1  , 512  , 7   , 7   ), (512 , 512  , 3  , 3  ), (1 , 1 , 1 , 1 ), ( 1 , 1 ), (1, 1), True  , True, False , [  7   , 512 , 49   , 32   , 512  ])),
            ("add_a_conv_run016", add_a_conv_run, (
            (1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False,
            [28, 256, 784, 16, 32])),
            ("add_a_conv_run017", add_a_conv_run, (
            (1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False, [28, 64, 784, 16, 32])),
            # ("add_a_conv_run018", add_a_conv_run,((1  , 64   , 56  , 56  ), (64  , 64   , 3  , 3  ), (1 , 1 , 1 , 1 ), ( 1 , 1 ), (1, 1), True  , False , False , [  9, 64   , 336  , 16   , 64   ])),
            # ("add_a_conv_run019", add_a_conv_run,((1  , 256  , 56  , 56  ), (512 , 256  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True, False , False , [  7   , 512  , 196  , 64   , 256  ])),
            # ("add_a_conv_run020", add_a_conv_run,((1  , 512  , 28  , 28  ), (1024, 512  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True, True, False , [  3  , 1024 , 112  , 32   , 512  ])),
        ]
        return

    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

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


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test2():
    a = TestCase()
    a.setup()
    a.test(3, 2)
    a.teardown()
