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
matmul addn trans data fusion
"""
import os
import pytest
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.matmul_addn_transdata_run import matmul_addn_transdata_execute


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_matmul_addn_transdata_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape_x, shape_y, bias, left_format, right_format, output_format, adj_x, adj_y, dtype, bias_dtype, out_dtype, kernel_name, attrs
            ("matmul_addn_transdata_0", matmul_addn_transdata_execute, ((4096, 768), (768, 12288), 0,  3, "zN", "zN", "zN", False, False, "float16", None, "float16", "matmul_addn_transdata_cce")),
            ("matmul_addn_transdata_1", matmul_addn_transdata_execute, ((4096, 768), (4096, 12288), 0,  2, "zN", "zN", "zN", True, False, "float16", None, "float32", "matmul_addn_transdata_cce")),
        ]

        self.testarg_rpc_cloud = [
        ]
        self.testarg_level1 = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape_x, shape_y, bias, left_format, right_format, output_format, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs
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
        self.common_run(self.testarg)

    def test_rpc_cloud(self):
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