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
import datetime
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.depthwise_ad_run import depthwise_ad_run


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

            # (testflag, opfuncname,(N, H, W, CI, k_ch, KH, KW, pad_h, pad_w, stride_h, stride_w, cutH, cutCo, cutM, cutK, cutN)
            # mobilenet V1
            ("depthwise_ad_run_001", depthwise_ad_run, (16, 7, 7, 1024, 1, 3, 3, 1, 1, 1, 1, 7, 16, 512, 3 * 16, 16)),

            # mobilenet V2
            ("depthwise_ad_run_101", depthwise_ad_run, (16, 7, 7, 960, 1, 3, 3, 1, 1, 1, 1, 9, 16, 512, 3 * 16, 16)),

            # (testflag, opfuncname,(N, H, H, CI, CO, group, K, K, pad, pad, stride, stride, cutH, cutCo, cutM, cutK, cutN)
            #("depthwise_ad_run_001", depthwise_ad_run, (16, 112, 112,  32,  32, 2 , 3, 3, 1, 1, 1, 1, 15, 16, 512, 3*16, 16)),
            # ("depthwise_ad_run_001", depthwise_ad_run, (32, 112, 112,  32,  32, 2, 3, 3, 1, 1, 1, 1, 16, 16, 16, 16, 16)),

            #("depthwise_ad_run_002", depthwise_ad_run,(16, 112, 112,  64,  64, 4 , 3, 3, 1, 1, 2, 2, 15, 16, 512, 3*16, 16)),

            #("depthwise_ad_run_003", depthwise_ad_run,(1,  56,  56, 128, 128, 8 , 3, 3, 1, 1, 1, 1, 15, 16, 512, 3*16, 16)),

            # ("depthwise_ad_run_004", depthwise_ad_run,(1,  56,  56, 128, 128, 8 , 3, 3, 1, 1, 2, 2, 15, 16, 512, 3*16, 16)),

            # ("depthwise_ad_run_005", depthwise_ad_run,(1,  28,  28, 256, 256, 16, 3, 3, 1, 1, 1, 1, 7, 16, 512, 3*16, 16)),

            # ("depthwise_ad_run_006", depthwise_ad_run,(1,  28,  28, 256, 256, 16, 3, 3, 1, 1, 2, 2, 7, 16, 512, 3*16, 16)),

            # ("depthwise_ad_run_007", depthwise_ad_run,(1,  14,  14, 512, 512, 32, 3, 3, 1, 1, 1, 1, 7, 16, 512, 3*16, 16)),

            # ("depthwise_ad_run_008", depthwise_ad_run,(1,  14,  14, 512, 512, 32, 3, 3, 1, 1, 2, 2, 7, 16, 512, 3*16, 16)),

            # ("depthwise_ad_run_009", depthwise_ad_run,(1,   7,   7,1024,1024, 64, 3, 3, 1, 1, 1, 1, 7, 16, 512, 3*16, 16)),

            # # mobilenet V2
            # ("depthwise_ad_run_101", depthwise_ad_run,(1, 112, 112,  32,  32, 2 , 3, 3, 1, 1, 1, 1, 114, 16, 640, 48, 16)),

            # ("depthwise_ad_run_102", depthwise_ad_run,(1, 112, 112,  96,  96, 6 , 3, 3, 1, 1, 2, 2, 114, 16, 432, 48, 16)),

            # ("depthwise_ad_run_103", depthwise_ad_run,(1,  56,  56, 144, 144, 9 , 3, 3, 1, 1, 1, 1, 58, 16, 176, 96, 16)),

            # ("depthwise_ad_run_104", depthwise_ad_run,(1,  56,  56, 144, 144, 9 , 3, 3, 1, 1, 2, 2, 58, 16, 14*16, 9*16, 16)),

            # ("depthwise_ad_run_105", depthwise_ad_run,(1,  28,  28, 192, 192, 12, 3, 3, 1, 1, 1, 1, 30, 16, 16*7, 9*16, 16)),

            # ("depthwise_ad_run_106", depthwise_ad_run,(1,  28,  28, 192, 192, 12, 3, 3, 1, 1, 2, 2, 30, 16, 512, 3*16, 16)),

            # ("depthwise_ad_run_107", depthwise_ad_run,(1,  14,  14, 384, 384, 24, 3, 3, 1, 1, 1, 1, 16, 16, 160, 80, 16)),

            # ("depthwise_ad_run_108", depthwise_ad_run,(1,  14,  14, 576, 576, 36, 3, 3, 1, 1, 1, 1, 16, 16, 208, 9*16, 16)),

            # ("depthwise_ad_run_109", depthwise_ad_run,(1,  14,  14, 576, 576, 36, 3, 3, 1, 1, 2, 2, 16, 16, 64, 9*16, 16)),

            # ("depthwise_ad_run_110", depthwise_ad_run,(1,   7,   7, 960, 960, 60, 3, 3, 1, 1, 1, 1, 9, 16, 512, 3*16, 16)),
        ]
        return

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


if __name__ == "__main__":
    #a = TestCase("test_depthwise_ad_001", os.getcwd())
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
