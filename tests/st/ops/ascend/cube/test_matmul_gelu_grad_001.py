# Copyright 2021 Huawei Technologies Co., Ltd
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
matmul gelu grad fusion
"""
import os
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.ascend.matmul_gelu_grad_run import matmul_gelu_grad_execute

class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_matmul_gelu_grad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape_x, shape_y, bias, left_format, right_format, output_format, adj_x, adj_y, dtype, bias_dtype, out_dtype, kernel_name, attrs
            ("matmul_gelu_grad_0", matmul_gelu_grad_execute, ((4096, 192), (4096, 12288), 0,  "zN", "zN", "zN", True, False, "float16", None, "float32", "matmul_gelu_grad_cce")),
            ("matmul_gelu_grad_1", matmul_gelu_grad_execute, ((4096, 12288), (768, 12288), 0, "zN", "zN", "zN", False, True, "float16", None, "float32", "matmul_gelu_grad_cce")),

        ]

        self.testarg_rpc_cloud = [
        ]
        self.testarg_level1 = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape_x, shape_y, bias, left_format, right_format, output_format, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs
        ]

        return

    def test_run(self):
        self.common_run(self.testarg)

    def test_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def test(self, split_nums, split_idx):
        self.common_run(get_splitted_cases(self.testarg, split_nums, split_idx))

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


def test0():
    a = TestCase()
    a.setup()
    a.test(2, 0)
    a.teardown()


def test1():
    a = TestCase()
    a.setup()
    a.test(2, 1)
    a.teardown()
