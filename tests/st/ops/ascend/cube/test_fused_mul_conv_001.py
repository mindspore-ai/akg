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


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_fused_mul_conv_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #("fused_mul_conv_run001", fused_mul_conv_run,((1  , 1024 , 14  , 14  ), (2048, 1024 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True  , True  , False)),
            #("fused_mul_conv_run002", fused_mul_conv_run,((1  , 1024 , 14  , 14  ), (256 , 1024 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , True  , False)),
            #("fused_mul_conv_run003", fused_mul_conv_run,((1  , 1024 , 14  , 14  ), (512 , 1024 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True  , True  , False)),
            #("fused_mul_conv_run004", fused_mul_conv_run,((1  , 128  , 28  , 28  ), (128 , 128  , 3  , 3  ), (1 , 1 , 1 , 1 ), ( 1 , 1 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run005", fused_mul_conv_run,((1  , 128  , 28  , 28  ), (512 , 128  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run006", fused_mul_conv_run,((1  , 2048 , 7   , 7   ), (512 , 2048 , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , True  , False)),
            #("fused_mul_conv_run007", fused_mul_conv_run,((1  , 256  , 14  , 14  ), (1024, 256  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run008", fused_mul_conv_run,((1  , 256  , 14  , 14  ), (256 , 256  , 3  , 3  ), (1 , 1 , 1 , 1 ), ( 1 , 1 ), (1, 1), True  , True  , False)),
            #("fused_mul_conv_run009", fused_mul_conv_run,((1  , 256  , 56  , 56  ), (128 , 256  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run010", fused_mul_conv_run,((1  , 256  , 56  , 56  ), (64  , 256  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run011", fused_mul_conv_run,((1  , 3    , 224 , 224 ), (64  , 3    , 7  , 7  ), (3 , 3 , 3 , 3 ), ( 2 , 2 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run012", fused_mul_conv_run,((1  , 512  , 28  , 28  ), (128 , 512  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run013", fused_mul_conv_run,((1  , 512  , 28  , 28  ), (256 , 512  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run014", fused_mul_conv_run,((1  , 512  , 7   , 7   ), (2048, 512  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , True  , False)),
            #("fused_mul_conv_run015", fused_mul_conv_run,((1  , 512  , 7   , 7   ), (512 , 512  , 3  , 3  ), (1 , 1 , 1 , 1 ), ( 1 , 1 ), (1, 1), True  , True  , False)),
            #("fused_mul_conv_run016", fused_mul_conv_run,((1  , 64   , 56  , 56  ), (256 , 64   , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run017", fused_mul_conv_run,((1  , 64   , 56  , 56  ), (64  , 64   , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 1 , 1 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run018", fused_mul_conv_run,((1  , 64   , 56  , 56  ), (64  , 64   , 3  , 3  ), (1 , 1 , 1 , 1 ), ( 1 , 1 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run019", fused_mul_conv_run,((1  , 256  , 56  , 56  ), (512 , 256  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True  , False , False)),
            #("fused_mul_conv_run020", fused_mul_conv_run,((1  , 512  , 28  , 28  ), (1024, 512  , 1  , 1  ), (0 , 0 , 0 , 0 ), ( 2 , 2 ), (1, 1), True  , True  , False)),
        ]

        self.testarg_level1 = [
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
    t = TestCase()
    t.setup()
    t.test_run()
