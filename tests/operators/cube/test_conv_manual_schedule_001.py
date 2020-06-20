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
convolution tensor compute op manual schedule test cases
"""

import datetime
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.conv_run_mansch import conv_run_mansch


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_conv_001"
        case_path = os.getcwd()
        max_retry = 0
        self.params_init(case_name, case_path, max_retry)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, 
            # testflag, opfuncname, FMap_shape, Filter_shape , Pad , Stride, Dilation, use_bias, bypass_l1, dump_data, Tile
            # (testflag, opfuncname, ((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (pad_top, pad_bottom, pad_left, pad_right), (stride_h, stride_w), (dilation_h, dilation_w), bias, bypass_l1 , dump_data, [cutH, cutCo, cutM, cutK, cutN]))
            ("conv_run_001", conv_run_mansch, ((1, 64, 33, 33), (16, 64, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), True, False, False, [5, 0, 16, 18 * 16, 16])),
            ("conv_run_002", conv_run_mansch, ((1, 64, 32, 32), (16, 64, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), True, False, False, [5, 0, 16, 36 * 16, 16])),
            # # # exceeds L1
            ("conv_run_003", conv_run_mansch, ((1, 96, 36, 36), (256, 96, 5, 5), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False, [20, 0, 16, 128, 16])),
            ("conv_run_004", conv_run_mansch, ((1, 64, 33, 33), (16, 64, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), True, False, False, [5, 0, 16, 36 * 16, 16])),
            ("conv_run_005", conv_run_mansch, ((1, 64, 31, 31), (16, 64, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), True, False, False, [5, 0, 16, 36 * 16, 16])),
            ("conv_run_006", conv_run_mansch, ((1, 64, 33, 33), (16, 64, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), True, False, False, [7, 0, 16, 36 * 16, 16])),
            ("conv_run_007", conv_run_mansch, ((1, 64, 31, 31), (16, 64, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), True, False, False, [7, 0, 16, 36 * 16, 16])),
            ("conv_run_008", conv_run_mansch, ((1, 64, 31, 33), (16, 64, 3, 3), (1, 1, 0, 0), (2, 2), (1, 1), True, False, False, [5, 0, 16, 36 * 16, 16])),
            ("conv_run_009", conv_run_mansch, ((1, 64, 33, 31), (16, 64, 3, 3), (0, 0, 1, 1), (2, 2), (1, 1), True, False, False, [5, 0, 16, 36 * 16, 16])),
            ("conv_run_010", conv_run_mansch, ((1, 64, 31, 33), (16, 64, 3, 3), (1, 1, 0, 0), (2, 2), (1, 1), True, False, False, [7, 0, 16, 36 * 16, 16])),
            ("conv_run_011", conv_run_mansch, ((1, 64, 33, 31), (16, 64, 3, 3), (0, 0, 1, 1), (2, 2), (1, 1), True, False, False, [7, 0, 16, 36 * 16, 16])),
            ("conv_run_012", conv_run_mansch, ((1, 64, 33, 33), (16, 64, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), True, False, False, [7, 0, 16, 128, 16])),
            ("conv_run_013", conv_run_mansch, ((1, 64, 31, 33), (16, 64, 3, 3), (1, 1, 0, 0), (2, 2), (1, 1), True, False, False, [7, 0, 16, 128, 16])),
            ("conv_run_014", conv_run_mansch, ((1, 64, 33, 31), (16, 64, 3, 3), (0, 0, 1, 1), (2, 2), (1, 1), True, False, False, [7, 0, 16, 128, 16])),
            ("conv_run_015", conv_run_mansch, ((1, 64, 31, 31), (16, 64, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), True, False, False, [7, 0, 16, 128, 16])),
            # # Wo = 2*16
            ("conv_run_016", conv_run_mansch, ((1, 16, 65, 65), (16, 16, 3, 3), (0, 0, 0, 0), (2, 2), (1, 1), True, False, False, [5, 0, 16, 9 * 16, 16])),
            # # Wo = 3*16
            ("conv_run_017", conv_run_mansch, ((1, 16, 50, 50), (16, 16, 3, 3), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False, [5, 0, 16, 9 * 16, 16])),
            # # resnet50
            ("conv_run_018", conv_run_mansch, ((1, 16, 224, 224), (64, 16, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1), True, False, False, [7, 0, 16, 128, 16])),
            # smaller version inspired by resnet50
            ("conv_run_019", conv_run_mansch, ((1, 64, 30, 30), (16, 64, 3, 3), (2, 2, 2, 2), (1, 1), (1, 1), True, False, False, [3, 0, 16, 16, 16])),
            ("conv_run_020", conv_run_mansch, ((1, 64, 56, 64), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False, [1, 0, 16, 16, 16])),
            ("conv_run_021", conv_run_mansch, ((1, 64, 56, 64), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True, False, False, [3, 0, 16, 16, 16])),
            ("conv_run_022", conv_run_mansch, ((1, 128, 28, 32), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True, False, False, [3, 0, 16, 72*16, 16])),
            # doesn't fit in L1 case
            ("conv_run_023", conv_run_mansch, ((1, 256, 14, 16), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True, False, False, [3, 0, 16, 16, 16])),
            # doesn't fit in L1 case
            ("conv_run_024", conv_run_mansch, ((1, 512, 7, 16), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), True, False, False, [3, 0, 16, 16, 16])),
            ("conv_run_025", conv_run_mansch, ((1, 64, 31, 31), (16, 64, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), True, False, False, [7, 0, 32, 128, 16])),
            ("conv_run_026", conv_run_mansch, ((1, 16, 224, 224), (16, 16, 7, 7), (3, 3, 3, 3), (2, 2), (1, 1), True, False, False, [7, 0, 16, 128, 16])),
            ("conv_run_027", conv_run_mansch, ((1, 64, 31, 31), (16, 64, 3, 3), (1, 1, 1, 1), (2, 2), (1, 1), True, False, False, [5, 0, 32, 128, 16])),
            ("conv_run_028", conv_run_mansch, ((1, 64, 7, 16), (16, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False, [2, 0, 16, 2*16, 16])),
            ("conv_run_029", conv_run_mansch, ((10, 64, 7, 16), (16, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), True, False, False, [2, 0, 16, 2*16, 16])),
            ("conv_run_030", conv_run_mansch, ((1, 16, 52, 35), (16, 16, 5, 5), (0, 0, 0, 0), (1, 2), (1, 1), True, False, False, [5, 0, 16, 16, 16])),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return

