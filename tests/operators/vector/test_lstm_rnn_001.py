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
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""

import datetime
import os

from base import TestBase
import pytest
from test_run.lstm_rnn_run import lstmcell_run
from test_run.lstm_rnn_run import rnn_tanh_cell_run
from test_run.lstm_rnn_run import rnn_relu_cell_run
from test_run.lstm_rnn_ad_run import lstmcell_grad_c_run, lstmcell_grad_h_run
from test_run.lstm_rnn_ad_run import rnn_tanh_cell_grad_run, rnn_relu_cell_grad_run, rnn_tanh_cell_ad_run, rnn_relu_cell_ad_run
from test_run.lstm_rnn_ad_run import lstmcell_h_ad_run, lstmcell_c_ad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_lstm_rnn"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # # Forward LSTMcell
            # Auto-dim
            ("lstm_rnn_0", lstmcell_run, ((1, 16, 16), "float16", "lstm_rnn_0_fp16")),
            ("lstm_rnn_1", lstmcell_run, ((2, 12, 64), "float16", "lstm_rnn_1_fp16"), ),
            ("lstm_rnn_2", lstmcell_run, ((1, 16, 32), "float16", "lstm_rnn_2_fp16")),
            ("lstm_rnn_3", lstmcell_run, ((1, 100, 100), "float16", "lstm_rnn_2_fp16")),

            # Manual dims
            ("lstm_rnn_0", lstmcell_run, ((2, 256, 64), "float16", "lstm_rnn_0_fp16"), ((1, 1), )),
            ("lstm_rnn_0", lstmcell_run, ((2, 16, 128), "float16", "lstm_rnn_0_fp16"), ((1, 1), )),
            ("lstm_rnn_0", lstmcell_run, ((1024, 16, 32), "float16", "lstm_rnn_0_fp16"), ((1, 1), )),
            ("lstm_rnn_0", lstmcell_run, ((2, 1024, 16), "float16", "lstm_rnn_0_fp16"), ((1, 1), )),
            ("lstm_rnn_0", lstmcell_run, ((128, 1024, 16), "float16", "lstm_rnn_0_fp16"), ((1, 1), (1, 1))),
            ("lstm_rnn_0", lstmcell_run, ((1024, 16, 64), "float16", "lstm_rnn_0_fp16"), ((1, 1), )),

            # # Forward RNN
            # Auto-dim
            ("lstm_rnn_1", rnn_tanh_cell_run, ((2, 16, 32), "float16", "lstm_rnn_1_fp16")),
            ("lstm_rnn_1", rnn_relu_cell_run, ((2, 16, 32), "float16", "lstm_rnn_1_fp16")),

            # Manual dims
            ("lstm_rnn_1", rnn_tanh_cell_run, ((2, 16, 32), "float16", "lstm_rnn_1_fp16"), ((1, 1), )),
            ("lstm_rnn_1", rnn_relu_cell_run, ((2, 16, 32), "float16", "lstm_rnn_1_fp16"), ((1, 1), )),

            # Grad RNN compiled but not tested numerically
            ("lstm_rnn_ad_1", rnn_relu_cell_grad_run, ((1, 16, 32), "float16", "lstm_rnn_1_fp16")),
            ("lstm_rnn_ad_1", rnn_tanh_cell_grad_run, ((1, 16, 32), "float16", "lstm_rnn_1_fp16")),

            # AD PASS all inputs
            ("lstm_rnn_ad_1", rnn_tanh_cell_ad_run, ((1, 16, 32), "float16", "lstm_rnn_ad_1_fp16"), ((96, 96),)),
            ("lstm_rnn_ad_2", rnn_relu_cell_ad_run, ((1, 16, 32), "float16", "lstm_rnn_ad_2_fp16")),
            ("lstm_rnn_ad_3", lstmcell_c_ad_run, ((1, 16, 32), "float16", "lstm_c_ad_3_fp16"), ((65536, 65536), )),
            # Long compilation time, run at level = 2
            # ("lstm_rnn_ad_4", lstmcell_h_ad_run, ((1, 16, 32), "float16", "lstm_h_ad_4_fp16")), ((65536, 65536), ),

            #############################################
            # Grad LSTM compilation failed in codegen.build_module
            # ("lstm_rnn_grad_1", lstmcell_grad_h_run, ((1, 16, 32), "float16", "lstm_rnn_grad_1_fp16"), ((1, 1), )),
            # ("lstm_rnn_grad_0", lstmcell_grad_c_run, ((1, 16, 32), "float16", "lstm_rnn_grad_0_fp16")),
            # Grad LSTM compilation failed in memory overflow
            # ("lstm_rnn_ad_4", lstmcell_c_ad_run, ((1, 100, 100), "float16", "lstm_c_ad_4_fp16")),

            # # Fail numerical result error
            # ("lstm_rnn_0", lstmcell_run, ((2, 16, 18), "float16", "lstm_rnn_0_fp16"), ((1, 1), )),
            # ("lstm_rnn_0", lstmcell_run, ((128, 1024, 16), "float16", "lstm_rnn_0_fp16")),
            # ("lstm_rnn_0", lstmcell_run, ((1024, 16, 64), "float16", "lstm_rnn_0_fp16")),
            # ("lstm_rnn_0", lstmcell_run, ((2, 257, 64), "float16", "lstm_rnn_0_fp16"))
            # ("lstm_rnn_0", lstmcell_run, ((64, 1024, 16), "float16", "lstm_rnn_0_fp16"))
            # ("lstm_rnn_0", lstmcell_run, ((2, 128, 128), "float16", "lstm_rnn_0_fp16"))
            # ("lstm_rnn_0", lstmcell_run, ((2, 16, 256), "float16", "lstm_rnn_0_fp16")),
            # ("lstm_rnn_ad_5", lstmcell_c_ad_run, ((1, 16, 100), "float16", "lstm_c_ad_4_fp16")),
            #############################################
        ]

        self.testarg_level2 = [
            # AD PASS only for W_ih, w_hh, b_ih, b_hh
            ("lstm_rnn_ad_3", lstmcell_c_ad_run, ((1, 16, 32), "float16", "lstm_c_ad_3_fp16"), ((65536, 65536), )),
            # Long compilation time
            ("lstm_rnn_ad_4", lstmcell_h_ad_run, ((1, 16, 32), "float16", "lstm_h_ad_4_fp16"), ((65536, 65536), )),
        ]

    @pytest.mark.rpc_mini
    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.level2
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return

# if __name__=="__main__":
#     t = TestCase()
#     t.setup()
#     t.test_run_level2()
