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
test_fused_minimum_or_maximum_grad
"""
import os
import pytest
from tests.common.base import TestBase


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_fused_minimum_or_maximum_grad"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        kernel = "fused_minimum_or_maximum_grad_run"
        kernel_name = "cce_min_max_grad_fp16"
        self.testarg = [
            ## testflag, opfuncname, testRunArgs, dimArgs
            # shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype, kernel_name, attrs
            ("fused_min_or_max_grad_01", kernel, ((64,), (64,), (64,), True, True, "GE", "float16", kernel_name)),
            ("fused_min_or_max_grad_02", kernel, ((64,), (64,), (64,), True, True, "LE", "float16", kernel_name)),
            ("fused_min_or_max_grad_03", kernel, ((64,), (64,), (64,), False, True, "GE", "float16", kernel_name)),
            ("fused_min_or_max_grad_04", kernel, ((64, 64), (64, 64), (64, 64), True, False, "GE", "float16", kernel_name)),
            ("fused_min_or_max_grad_05", kernel, ((128,), (128,), (128,), True, False, "LE", "float16", kernel_name)),
            ("fused_min_or_max_grad_06", kernel, ((128,), (128,), (128,), False, True, "LE", "float16", kernel_name)),
            ("fused_min_or_max_grad_07", kernel, ((64,), (64,), (1,), True, True, "GE", "float16", kernel_name)),
            ("fused_min_or_max_grad_08", kernel, ((64,), (1,), (64,), True, True, "GE", "float16", kernel_name)),
        ]
        self.testarg_cloud = [
            ## testflag, opfuncname, testRunArgs, dimArgs
            # shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype, kernel_name, attrs
            ("fused_min_or_max_grad_01", kernel, ((64,), (64,), (64,), True, True, "GE", "float32", kernel_name)),
            ("fused_min_or_max_grad_01", kernel, ((2,), (2,), (2,), True, True, "GE", "int32", kernel_name)),
            # failed
            #('fused_min_or_max_grad_01', kernel, ((2, 2, 3, 2), (3, 1), (2, 2, 1, 2), True, True, 'GE', 'float32', kernel_name)),
        ]

        self.testarg_rpc_cloud = [
            ## testflag, opfuncname, testRunArgs, dimArgs
            # shape_dz, shape_x, shape_y, grad_x, grad_y, op_type, dtype, kernel_name, attrs
            ("fused_min_or_max_grad_02_fp32", kernel, ((1,), (1,), (1,), True, True, "GE", "float32", kernel_name)),
            ("fused_min_or_max_grad_03_int32", kernel, ((2,), (2,), (1,), True, True, "GE", "int32", kernel_name)),
            ("fused_min_or_max_grad_04_fp32", kernel, ((128, 128, 64), (128, 128, 64), (1,), True, True, "GE", "float32", kernel_name)),
            ("fused_min_or_max_grad_05_fp32", kernel, ((64, 128, 768), (64, 128, 768), (1,), True, True, "GE", "float32", kernel_name)),
            ("fused_min_or_max_grad_06_fp32", kernel, ((1, 1), (1, 1), (1,), True, True, "GE", "float32", kernel_name)),
            ("fused_min_or_max_grad_07_int32", kernel, ((3,), (3,), (1,), True, True, "GE", "int32", kernel_name)),
            ("fused_min_or_max_grad_08_fp32", kernel, ((1,), (1,), (1,), True, True, "LE", "float32", kernel_name)),
            ("fused_min_or_max_grad_09_fp32", kernel, ((128, 128, 64), (128, 128, 64), (1,), True, True, "LE", "float32", kernel_name)),
            ("fused_min_or_max_grad_10_fp32", kernel, ((64, 128, 768), (64, 128, 768), (1,), True, True, "LE", "float32", kernel_name)),

            # ci random fail ("fused_min_or_max_grad_01", kernel, ((64,), (64,), (64,), True, True, "GE", "float16", kernel_name)),
            ("fused_min_or_max_grad_02", kernel, ((64,), (64,), (64,), True, True, "LE", "float16", kernel_name)),
            ("fused_min_or_max_grad_03", kernel, ((64,), (64,), (64,), False, True, "GE", "float16", kernel_name)),
            ("fused_min_or_max_grad_04", kernel, ((64, 64), (64, 64), (64, 64), True, False, "GE", "float16", kernel_name)),
            ("fused_min_or_max_grad_05", kernel, ((128,), (128,), (128,), True, False, "LE", "float16", kernel_name)),
            ("fused_min_or_max_grad_06", kernel, ((128,), (128,), (128,), False, True, "LE", "float16", kernel_name)),
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

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
