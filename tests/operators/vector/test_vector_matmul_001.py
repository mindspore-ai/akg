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
test for this case set cloud
def launch(kernel, args, output = (-1,), kernel_meta_path = './kernel_meta', spec=Spec.CLOUD):

vector matmul is a special operator for cloud, the data type of
the input tensor is float32. dataflow is DDR -> UB -> DDR
use vector instric to implement matmul in bert network.

There is a special case in bert is

left : (M,K) : (512, 2) : float32
right: (K,N) : (2,1024) : float32

all case about float32 matmul is:
            index  left_matrix | right_matrix    test
transpose      0       False    |  False         pass
transpose      1       False    |  True          pass
transpose      2       False    |  True          fail by code gen
transpose      3       True     |  True          fail by no intrinic

"""
import datetime
import os

from base import TestBase
import pytest
from test_run.vector_matmul_run import vector_matmul_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_vector_matmul_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            #case_index, m, n, k, trans_a, trans_b, read_data, dump_data, dtype, kernel_name, attrs

            # trans_a = False, trans_b = False
            ("vector_matmul_run1", vector_matmul_run, (0, 48, 32, 64, False, False, False, False, "float32", "matmul_fp32")),
            # trans_a = False, trans_b = False
            #("vector_matmul_run2", vector_matmul_run,(1,1024,1024,2,False, False, False, False, "float32", "matmul_fp32")),
            # # trans_a = True, trans_b = False
            #("vector_matmul_run3", vector_matmul_run,(2, 48, 32, 64, True, False, False, False, "float32", "matmul_fp32"),((16, 8),(16, 8))),
            # """
            # trans_a = False, trans_b = True

            # This is the failed case, caused by codegen reduce by the last axis
            # """
            # ("vector_matmul_run4", vector_matmul_run,
            # (3, 48, 32, 64, False, True, False, False, "float32", "matmul_fp32"),
            # ((12, 8),(32, 8))),


            # ("matmul_run33", matmul_run,
            ## (33, (1,), 2, 1024, 8192, 0, 0, False, True, False, True, "float16", "float16", "matmul_cce"),
            # ((4, 4), (16, 16), (16, 16), (16, 16))),
        ]
        self.testarg_1 = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            #case_index, m, n, k, trans_a, trans_b, read_data, dump_data, dtype, kernel_name, attrs

            # trans_a = False, trans_b = False
            # ("vector_matmul_run1",vector_matmul_run,(0,48,32,64,False,False,False,False,"float32","matmul_fp32")),
            # trans_a = False, trans_b = False
            ("vector_matmul_run2", vector_matmul_run, (1, 1024, 1024, 2, False, False, False, False, "float32", "matmul_fp32")),
            # # trans_a = True, trans_b = False
            ("vector_matmul_run3", vector_matmul_run, (2, 48, 32, 64, True, False, False, False, "float32", "matmul_fp32"), ((16, 8), (16, 8))),
            # """
            # trans_a = False, trans_b = True

            # This is the failed case, caused by codegen reduce by the last axis
            # """
            # ("vector_matmul_run4", vector_matmul_run,
            # (3, 48, 32, 64, False, True, False, False, "float32", "matmul_fp32"),
            # ((12, 8),(32, 8))),


            # ("matmul_run33", matmul_run,
            ## (33, (1,), 2, 1024, 8192, 0, 0, False, True, False, True, "float16", "float16", "matmul_cce"),
            # ((4, 4), (16, 16), (16, 16), (16, 16))),
        ]

        self.testarg_rpc_cloud = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape_x, shape_y, bias, bypass, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs
            ("vector_matmul_run10", vector_matmul_run, (0, 160, 1024, 1024, False, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run13", vector_matmul_run, (1, 64, 2, 1024, False, True, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run14", vector_matmul_run, (1, 1024, 1024, 2, False, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run18", vector_matmul_run, (1, 8, 2, 1024, False, True, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run20", vector_matmul_run, (2, 1024, 1024, 1280, True, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run22", vector_matmul_run, (1, 8192, 1024, 2, False, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run24", vector_matmul_run, (1, 64, 1024, 1024, False, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run25", vector_matmul_run, (2, 2, 1024, 64, True, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run31", vector_matmul_run, (0, 8, 1024, 1024, False, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run32", vector_matmul_run, (0, 1024, 1024, 1024, False, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run33", vector_matmul_run, (0, 64, 1024, 2, False, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run35", vector_matmul_run, (0, 1280, 1024, 1024, False, False, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run26", vector_matmul_run, (1, 160, 30522, 1024, False, True, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run34", vector_matmul_run, (1, 1280, 30522, 1024, False, True, False, False, "float32", "matmul_fp32")),
            #("vector_matmul_run16", vector_matmul_run, (1, 2, 1024, 8192, True, False, False, False, "float32", "matmul_fp32"), ((2, 2), (32, 32), (32, 32))),
            #("vector_matmul_run11", vector_matmul_run, (0, 8192, 1024, 4096, False, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run12", vector_matmul_run, (2, 1024, 1024, 8192, True, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run15", vector_matmul_run, (1, 1024, 1024, 4096, False, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run19", vector_matmul_run, (1, 8192, 1024, 1024, False, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run21", vector_matmul_run, (0, 1024, 4096, 1024, False, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run23", vector_matmul_run, (1, 1024, 4096, 8192, True, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run28", vector_matmul_run, (2, 4096, 1024, 8192, True, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run30", vector_matmul_run, (1, 8192, 1024, 4096, False, True, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run36", vector_matmul_run, (0, 8192, 4096, 1024, False, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #("vector_matmul_run37", vector_matmul_run, (1, 8192, 4096, 1024, False, True, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),

            #  cost a long time
            #  3857s for below this
            # ("vector_matmul_run17", vector_matmul_run, (1, 30522, 1024, 1280, True, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),
            #  3545s for below this
            # ("vector_matmul_run29", vector_matmul_run, (0, 1280, 1024, 30522, False, False, False, False, "float32", "matmul_fp32"), ((32, 32), (32, 32), (32, 32))),

            #  fait for now,
            #  As do not support that trans_a and trans_b both true:
            # ("vector_matmul_run27", vector_matmul_run, (3, 1024, 1024, 64, True, True, False, False, "float32", "matmul_fp32")),
        ]

        return

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.aicmodel1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_runi_night(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_1)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
