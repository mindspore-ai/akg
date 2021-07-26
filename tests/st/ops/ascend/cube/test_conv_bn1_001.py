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

"""testcase for conv_bn1 op"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.conv_bn1_run import conv_bn1_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_conv_bn1_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs,
            # testflag, opfuncname, case_num, fmap_shape               , filter_shape          , pad_                               , stride_    , dilation_  , use_bias, bypass_l1 , dump_data, Tile
            #(testflag, opfuncname, case_num, ((in_n, in_c, in_h, in_w), (cout, in_c, w_h, w_w), (p_left, p_right, p_top, p_bottom), (s_h, s_w), (d_h, d_w), bias    , bypass_l1 , dump_data, [cutH, cutCo, cutM, cutK, cutN]))
            ("conv_bn1_run001", conv_bn1_run, ((1, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run002", conv_bn1_run, ((1, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
        ]
        self.testarg_rpc_cloud= [
            ("conv_bn1_run003", conv_bn1_run, ((1, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run004", conv_bn1_run, ((1, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_bn1_run005", conv_bn1_run, ((1, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run006", conv_bn1_run, ((1, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run007", conv_bn1_run, ((1, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run008", conv_bn1_run, ((1, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_bn1_run009", conv_bn1_run, ((1, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run010", conv_bn1_run, ((1, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run011", conv_bn1_run, ((1, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)),
            ("conv_bn1_run012", conv_bn1_run, ((1, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run013", conv_bn1_run, ((1, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run014", conv_bn1_run, ((1, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run015", conv_bn1_run, ((1, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_bn1_run016", conv_bn1_run, ((1, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run017", conv_bn1_run, ((1, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run018", conv_bn1_run, ((1, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_bn1_run019", conv_bn1_run, ((1, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run020", conv_bn1_run, ((1, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run021", conv_bn1_run, ((1, 256, 56, 56), ( 128,  256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            ("conv_bn1_run022", conv_bn1_run, ((1, 512, 28, 28), ( 256,  512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            ("conv_bn1_run023", conv_bn1_run, ((1,1024, 14, 14), ( 512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            ("conv_bn1_run024", conv_bn1_run, ((1, 128, 56, 56), ( 128,  128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
            ("conv_bn1_run025", conv_bn1_run, ((1, 256, 28, 28), ( 256,  256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
            ("conv_bn1_run026", conv_bn1_run, ((1, 512, 14, 14), ( 512,  512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5

            ("conv_bn1_run001", conv_bn1_run, ((2, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run002", conv_bn1_run, ((2, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run003", conv_bn1_run, ((2, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run004", conv_bn1_run, ((2, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_bn1_run005", conv_bn1_run, ((2, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run006", conv_bn1_run, ((2, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run007", conv_bn1_run, ((2, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run008", conv_bn1_run, ((2, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_bn1_run009", conv_bn1_run, ((2, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run010", conv_bn1_run, ((2, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run011", conv_bn1_run, ((2, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)),
            ("conv_bn1_run012", conv_bn1_run, ((2, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run013", conv_bn1_run, ((2, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run014", conv_bn1_run, ((2, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run015", conv_bn1_run, ((2, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_bn1_run016", conv_bn1_run, ((2, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run017", conv_bn1_run, ((2, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            ("conv_bn1_run018", conv_bn1_run, ((2, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            ("conv_bn1_run019", conv_bn1_run, ((2, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run020", conv_bn1_run, ((2, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            ("conv_bn1_run021", conv_bn1_run, ((2, 256, 56, 56), ( 128,  256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            ("conv_bn1_run022", conv_bn1_run, ((2, 512, 28, 28), ( 256,  512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            ("conv_bn1_run023", conv_bn1_run, ((2,1024, 14, 14), ( 512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            ("conv_bn1_run024", conv_bn1_run, ((2, 128, 56, 56), ( 128,  128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
            ("conv_bn1_run025", conv_bn1_run, ((2, 256, 28, 28), ( 256,  256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
            ("conv_bn1_run026", conv_bn1_run, ((2, 512, 14, 14), ( 512,  512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
        ]
        self.testarg_level1 = [
            #("conv_bn1_run001", conv_bn1_run, ((32, 1024, 14, 14), (2048, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            #("conv_bn1_run002", conv_bn1_run, ((32, 1024, 14, 14), (256, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run003", conv_bn1_run, ((32, 1024, 14, 14), (512, 1024, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            #("conv_bn1_run004", conv_bn1_run, ((32, 128, 28, 28), (128, 128, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            #("conv_bn1_run005", conv_bn1_run, ((32, 128, 28, 28), (512, 128, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run006", conv_bn1_run, ((32, 2048, 7, 7), (512, 2048, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run007", conv_bn1_run, ((32, 256, 14, 14), (1024, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run008", conv_bn1_run, ((32, 256, 14, 14), (256, 256, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            #("conv_bn1_run009", conv_bn1_run, ((32, 256, 56, 56), (128, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            #("conv_bn1_run010", conv_bn1_run, ((32, 256, 56, 56), (64, 256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run011", conv_bn1_run, ((32, 3, 224, 224), (64, 3, 7, 7), (2, 3, 2, 3), (2, 2), (1, 1), False)),
            #("conv_bn1_run012", conv_bn1_run, ((32, 512, 28, 28), (128, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run013", conv_bn1_run, ((32, 512, 28, 28), (256, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            #("conv_bn1_run014", conv_bn1_run, ((32, 512, 7, 7), (2048, 512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run015", conv_bn1_run, ((32, 512, 7, 7), (512, 512, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            #("conv_bn1_run016", conv_bn1_run, ((32, 64, 56, 56), (256, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run017", conv_bn1_run, ((32, 64, 56, 56), (64, 64, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),
            #("conv_bn1_run018", conv_bn1_run, ((32, 64, 56, 56), (64, 64, 3, 3), (1, 1, 1, 1), (1, 1), (1, 1), False)),
            #("conv_bn1_run019", conv_bn1_run, ((32, 256, 56, 56), (512, 256, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            #("conv_bn1_run020", conv_bn1_run, ((32, 512, 28, 28), (1024, 512, 1, 1), (0, 0, 0, 0), (2, 2), (1, 1), False)),
            #("conv_bn1_run021", conv_bn1_run, ((32, 256, 56, 56), ( 128,  256, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            #("conv_bn1_run022", conv_bn1_run, ((32, 512, 28, 28), ( 256,  512, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            #("conv_bn1_run023", conv_bn1_run, ((32,1024, 14, 14), ( 512, 1024, 1, 1), (0, 0, 0, 0), (1, 1), (1, 1), False)),#1.5
            #("conv_bn1_run024", conv_bn1_run, ((32, 128, 56, 56), ( 128,  128, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
            #("conv_bn1_run025", conv_bn1_run, ((32, 256, 28, 28), ( 256,  256, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
            #("conv_bn1_run026", conv_bn1_run, ((32, 512, 14, 14), ( 512,  512, 3, 3), (0, 1, 0, 1), (2, 2), (1, 1), False)),#1.5
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
        self.common_run(self.testarg, is_conv=True)

    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1, is_conv=True)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
    a.test_run_rpc_cloud()
    a.test_run_level1()
    a.teardown()
