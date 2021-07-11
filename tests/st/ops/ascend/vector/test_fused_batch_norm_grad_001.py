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
fused_batch_norm_grad
"""
import os
import pytest
from tests.common.test_run.fused_batch_norm_grad_run import fused_batch_norm_grad_run
from tests.common.test_run.fused_batch_norm_grad_run import fused_bn_grad_5D_all_run
from tests.common.test_run.fused_batch_norm_grad_run import fused_bn_grad_5D_run_1
from tests.common.test_run.fused_batch_norm_grad_run import fused_bn_grad_5D_run_2
from tests.common.test_run.fused_batch_norm_grad_run import fused_bn_grad_5D_run_3
from tests.common.base import TestBase


class TestCase(TestBase):

    def setup(self):
        case_name = "test_auto_fused_tensor_batch_norm_grad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #  # caseflag, opfuncname, (shape, dtype, momentum, eps, data_format, axis, kerne_name)
            # ("Bng2dFp32_01", fused_batch_norm_grad_run, ((4, 512), "float32", 1e-5, "DefaultFormat", 1, "Bng2dFp32_01")),
            # ("Bng5dFp32_02", fused_batch_norm_grad_run, ((4, 4, 7, 7, 16), "float32", 1e-5, "NC1HWC0", None, "Bng5dFp32_02")),
            ("resnet50_Bng5dFp32_09", fused_bn_grad_5D_all_run, ((32, 8, 56, 56, 16), "float32", 1e-5, "resnet50_Bng5dFp32_09")),
            ("resnet50_Bng5dFp32_09", fused_bn_grad_5D_all_run, ((32, 16, 28, 28, 16), "float32", 1e-5, "resnet50_Bng5dFp32_09")),
            ("resnet50_Bng5dFp32_09", fused_bn_grad_5D_all_run, ((32, 32, 14, 14, 16), "float32", 1e-5, "resnet50_Bng5dFp32_09")),
            ("resnet50_Bng5dFp32_09", fused_bn_grad_5D_all_run, ((32, 8, 56, 56, 16), "float16", 1e-5, "resnet50_Bng5dFp32_09")),
            ("resnet50_Bng5dFp32_09", fused_bn_grad_5D_all_run, ((32, 16, 28, 28, 16), "float16", 1e-5, "resnet50_Bng5dFp32_09")),
            ("resnet50_Bng5dFp32_09", fused_bn_grad_5D_all_run, ((32, 32, 14, 14, 16), "float16", 1e-5, "resnet50_Bng5dFp32_09")),
        ]
        self.testarg_aic_cloud = [
            # ("Bng5dFp32_01", fused_batch_norm_grad_run, ((4, 4, 4, 4, 16), "float32", 1e-5, "NC1HWC0", None, "Bng5dFp32_01")),
        ]
        self.testarg_cloud = [
            # 2D
            # ("resnet101_Bng2dFp16_01", fused_batch_norm_grad_run, (( 64, 512), "float16", 1e-5, "DefaultFormat", 1, "resnet101_Bng2dFp16_01")),
            # ("resnet101_Bng2dFp16_02", fused_batch_norm_grad_run, (( 96, 512), "float16", 1e-5, "DefaultFormat", 1, "resnet101_Bng2dFp16_02")),
            # ("resnet101_Bng2dFp16_03", fused_batch_norm_grad_run, ((128, 512), "float16", 1e-5, "DefaultFormat", 1, "resnet101_Bng2dFp16_03")),
            # ("resnet101_Bng2dFp32_01", fused_batch_norm_grad_run, ((64, 512), "float32", 1e-5, "DefaultFormat", 1, "resnet101_Bng2dFp32_01")),
            # ("resnet101_Bng2dFp32_02", fused_batch_norm_grad_run, ((96, 512), "float32", 1e-5, "DefaultFormat", 1, "resnet101_Bng2dFp32_02")),
            # ("resnet101_Bng2dFp32_03", fused_batch_norm_grad_run, ((128, 512), "float32", 1e-5, "DefaultFormat", 1, "resnet101_Bng2dFp32_03")),

            # 5D
            # ("resnet50_Bng5dFp16_01",  fused_batch_norm_grad_run, ((32,   4, 112, 112, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_01")),
            # ("resnet50_Bng5dFp16_02",  fused_batch_norm_grad_run, ((32,   4,  56,  56, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_02")),
            # ("resnet50_Bng5dFp16_03",  fused_batch_norm_grad_run, ((32,  16,  56,  56, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_03")),
            # ("resnet50_Bng5dFp16_04",  fused_batch_norm_grad_run, ((32,   8,  28,  28, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_04")),
            # ("resnet50_Bng5dFp16_05",  fused_batch_norm_grad_run, ((32,  32,  28,  28, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_05")),
            # ("resnet50_Bng5dFp16_06",  fused_batch_norm_grad_run, ((32,  16,  14,  14, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_06")),
            # ("resnet50_Bng5dFp16_07",  fused_batch_norm_grad_run, ((32,  64,  14,  14, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_07")),
            # ("resnet50_Bng5dFp16_08",  fused_batch_norm_grad_run, ((32,  32,   7,   7, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_08")),
            # ("resnet50_Bng5dFp16_09",  fused_batch_norm_grad_run, ((32, 128,   7,   7, 16), "float16", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp16_09")),
            # ("resnet101_Bng5dFp16_10", fused_batch_norm_grad_run, ((32,   8,  56,  56, 16), "float16", 1e-5, "NC1HWC0", None, "resnet101_Bng5dFp16_10")),
            # ("resnet101_Bng5dFp16_11", fused_batch_norm_grad_run, ((32,  16,  28,  28, 16), "float16", 1e-5, "NC1HWC0", None, "resnet101_Bng5dFp16_11")),
            # ("resnet101_Bng5dFp16_12", fused_batch_norm_grad_run, ((32,  32,  14,  14, 16), "float16", 1e-5, "NC1HWC0", None, "resnet101_Bng5dFp16_12")),

            # ("resnet50_Bng5dFp32_01", fused_batch_norm_grad_run, ((32, 4, 112, 112, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_01")),
            # ("resnet50_Bng5dFp32_02", fused_batch_norm_grad_run, ((32, 4, 56, 56, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_02")),
            # ("resnet50_Bng5dFp32_03", fused_batch_norm_grad_run, ((32, 16, 56, 56, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_03")),
            # ("resnet50_Bng5dFp32_04", fused_batch_norm_grad_run, ((32, 8, 28, 28, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_04")),
            # ("resnet50_Bng5dFp32_05", fused_batch_norm_grad_run, ((32, 32, 28, 28, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_05")),
            # ("resnet50_Bng5dFp32_06", fused_batch_norm_grad_run, ((32, 16, 14, 14, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_06")),
            # ("resnet50_Bng5dFp32_07", fused_batch_norm_grad_run, ((32, 64, 14, 14, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_07")),
            # ("resnet50_Bng5dFp32_08", fused_batch_norm_grad_run, ((32, 32, 7, 7, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_08")),
            # ("resnet50_Bng5dFp32_09", fused_batch_norm_grad_run, ((32, 128, 7, 7, 16), "float32", 1e-5, "NC1HWC0", None, "resnet50_Bng5dFp32_09")),
            # ("resnet101_Bng5dFp32_10", fused_batch_norm_grad_run, ((32, 8, 56, 56, 16), "float32", 1e-5, "NC1HWC0", None, "resnet101_Bng5dFp32_10")),
            # ("resnet101_Bng5dFp32_11", fused_batch_norm_grad_run, ((32, 16, 28, 28, 16), "float32", 1e-5, "NC1HWC0", None, "resnet101_Bng5dFp32_11")),
            # ("resnet101_Bng5dFp32_12", fused_batch_norm_grad_run, ((32, 32, 14, 14, 16), "float32", 1e-5, "NC1HWC0", None, "resnet101_Bng5dFp32_12")),
        ]
        self.test_5D_split = [
            ## run all
            # resnet50 V.10
            ("resnet50_Bng5dFp32_01", fused_bn_grad_5D_all_run, ((32, 4, 112, 112, 16), "float32", 1e-5, "resnet50_Bng5dFp32_01")),
            ("resnet50_Bng5dFp32_02", fused_bn_grad_5D_all_run, ((32, 4, 56, 56, 16), "float32", 1e-5, "resnet50_Bng5dFp32_02")),
            ("resnet50_Bng5dFp32_03", fused_bn_grad_5D_all_run, ((32, 16, 56, 56, 16), "float32", 1e-5, "resnet50_Bng5dFp32_03")),
            ("resnet50_Bng5dFp32_04", fused_bn_grad_5D_all_run, ((32, 8, 28, 28, 16), "float32", 1e-5, "resnet50_Bng5dFp32_04")),
            ("resnet50_Bng5dFp32_05", fused_bn_grad_5D_all_run, ((32, 32, 28, 28, 16), "float32", 1e-5, "resnet50_Bng5dFp32_05")),
            ("resnet50_Bng5dFp32_06", fused_bn_grad_5D_all_run, ((32, 16, 14, 14, 16), "float32", 1e-5, "resnet50_Bng5dFp32_06")),
            ("resnet50_Bng5dFp32_07", fused_bn_grad_5D_all_run, ((32, 64, 14, 14, 16), "float32", 1e-5, "resnet50_Bng5dFp32_07")),
            ("resnet50_Bng5dFp32_08", fused_bn_grad_5D_all_run, ((32, 32, 7, 7, 16), "float32", 1e-5, "resnet50_Bng5dFp32_08")),
            ("resnet50_Bng5dFp32_09", fused_bn_grad_5D_all_run, ((32, 128, 7, 7, 16), "float32", 1e-5, "resnet50_Bng5dFp32_09")),
            ("resnet50_Bng5dFp16_01", fused_bn_grad_5D_all_run, ((32, 4, 112, 112, 16), "float16", 1e-5, "resnet50_Bng5dFp16_01")),
            ("resnet50_Bng5dFp16_02", fused_bn_grad_5D_all_run, ((32, 4, 56, 56, 16), "float16", 1e-5, "resnet50_Bng5dFp16_02")),
            ("resnet50_Bng5dFp16_03", fused_bn_grad_5D_all_run, ((32, 16, 56, 56, 16), "float16", 1e-5, "resnet50_Bng5dFp16_03")),
            ("resnet50_Bng5dFp16_04", fused_bn_grad_5D_all_run, ((32, 8, 28, 28, 16), "float16", 1e-5, "resnet50_Bng5dFp16_04")),
            ("resnet50_Bng5dFp16_05", fused_bn_grad_5D_all_run, ((32, 32, 28, 28, 16), "float16", 1e-5, "resnet50_Bng5dFp16_05")),
            ("resnet50_Bng5dFp16_06", fused_bn_grad_5D_all_run, ((32, 16, 14, 14, 16), "float16", 1e-5, "resnet50_Bng5dFp16_06")),
            ("resnet50_Bng5dFp16_07", fused_bn_grad_5D_all_run, ((32, 64, 14, 14, 16), "float16", 1e-5, "resnet50_Bng5dFp16_07")),
            ("resnet50_Bng5dFp16_08", fused_bn_grad_5D_all_run, ((32, 32, 7, 7, 16), "float16", 1e-5, "resnet50_Bng5dFp16_08")),
            ("resnet50_Bng5dFp16_09", fused_bn_grad_5D_all_run, ((32, 128, 7, 7, 16), "float16", 1e-5, "resnet50_Bng5dFp16_09")),

            #resnet50 V1.5 added
            ("resnet50_Bng5dFp32_10", fused_bn_grad_5D_all_run, ((32, 8, 56, 56, 16), "float32", 1e-5, "resnet50_Bng5dFp32_10")),
            ("resnet50_Bng5dFp32_11", fused_bn_grad_5D_all_run, ((32, 16, 28, 28, 16), "float32", 1e-5, "resnet50_Bng5dFp32_11")),
            ("resnet50_Bng5dFp32_12", fused_bn_grad_5D_all_run, ((32, 32, 14, 14, 16), "float32", 1e-5, "resnet50_Bng5dFp32_12")),
            ("resnet50_Bng5dFp16_10", fused_bn_grad_5D_all_run, ((32, 8, 56, 56, 16), "float16", 1e-5, "resnet50_Bng5dFp16_10")),
            ("resnet50_Bng5dFp16_11", fused_bn_grad_5D_all_run, ((32, 16, 28, 28, 16), "float16", 1e-5, "resnet50_Bng5dFp16_11")),
            ("resnet50_Bng5dFp16_12", fused_bn_grad_5D_all_run, ((32, 32, 14, 14, 16), "float16", 1e-5, "resnet50_Bng5dFp16_12")),

            ## run step 1
            # ("resnet50_Bng5dFp16_01_step1", fused_bn_grad_5D_run_1, ((32, 4, 112, 112, 16), "float16", "resnet50_Bng5dFp16_01")),
            # ("resnet50_Bng5dFp16_02_step1", fused_bn_grad_5D_run_1, ((32, 4, 56, 56, 16), "float16", "resnet50_Bng5dFp16_02")),
            # ("resnet50_Bng5dFp16_03_step1", fused_bn_grad_5D_run_1, ((32, 16, 56, 56, 16), "float16", "resnet50_Bng5dFp16_03")),
            # ("resnet50_Bng5dFp16_04_step1", fused_bn_grad_5D_run_1, ((32, 8, 28, 28, 16), "float16", "resnet50_Bng5dFp16_04")),
            # ("resnet50_Bng5dFp16_05_step1", fused_bn_grad_5D_run_1, ((32, 32, 28, 28, 16), "float16", "resnet50_Bng5dFp16_05")),
            # ("resnet50_Bng5dFp16_06_step1", fused_bn_grad_5D_run_1, ((32, 16, 14, 14, 16), "float16", "resnet50_Bng5dFp16_06")),
            # ("resnet50_Bng5dFp16_07_step1", fused_bn_grad_5D_run_1, ((32, 64, 14, 14, 16), "float16", "resnet50_Bng5dFp16_07")),
            # ("resnet50_Bng5dFp16_08_step1", fused_bn_grad_5D_run_1, ((32, 32, 7, 7, 16), "float16", "resnet50_Bng5dFp16_08")),
            # ("resnet50_Bng5dFp16_09_step1", fused_bn_grad_5D_run_1, ((32, 128, 7, 7, 16), "float16", "resnet50_Bng5dFp16_09")),

            ## run step 2
            # ("resnet50_Bng5dFp16_01_step2", fused_bn_grad_5D_run_2, ((32, 4, 112, 112, 16), "float16", 1e-5, "resnet50_Bng5dFp16_01")),
            # ("resnet50_Bng5dFp16_02_step2", fused_bn_grad_5D_run_2, ((32, 4, 56, 56, 16), "float16", 1e-5, "resnet50_Bng5dFp16_02")),
            # ("resnet50_Bng5dFp16_03_step2", fused_bn_grad_5D_run_2, ((32, 16, 56, 56, 16), "float16", 1e-5, "resnet50_Bng5dFp16_03")),
            # ("resnet50_Bng5dFp16_04_step2", fused_bn_grad_5D_run_2, ((32, 8, 28, 28, 16), "float16", 1e-5, "resnet50_Bng5dFp16_04")),
            # ("resnet50_Bng5dFp16_05_step2", fused_bn_grad_5D_run_2, ((32, 32, 28, 28, 16), "float16", 1e-5, "resnet50_Bng5dFp16_05")),
            # ("resnet50_Bng5dFp16_06_step2", fused_bn_grad_5D_run_2, ((32, 16, 14, 14, 16), "float16", 1e-5, "resnet50_Bng5dFp16_06")),
            # ("resnet50_Bng5dFp16_07_step2", fused_bn_grad_5D_run_2, ((32, 64, 14, 14, 16), "float16", 1e-5, "resnet50_Bng5dFp16_07")),
            # ("resnet50_Bng5dFp16_08_step2", fused_bn_grad_5D_run_2, ((32, 32, 7, 7, 16), "float16", 1e-5, "resnet50_Bng5dFp16_08")),
            # ("resnet50_Bng5dFp16_09_step2", fused_bn_grad_5D_run_2, ((32, 128, 7, 7, 16), "float16", 1e-5, "resnet50_Bng5dFp16_09")),

            ## run step 3
            # ("resnet50_Bng5dFp16_01_step3", fused_bn_grad_5D_run_3, ((32, 4, 112, 112, 16), "float16", "resnet50_Bng5dFp16_01")),
            # ("resnet50_Bng5dFp16_02_step3", fused_bn_grad_5D_run_3, ((32, 4, 56, 56, 16), "float16", "resnet50_Bng5dFp16_02")),
            # ("resnet50_Bng5dFp16_03_step3", fused_bn_grad_5D_run_3, ((32, 16, 56, 56, 16), "float16", "resnet50_Bng5dFp16_03")),
            # ("resnet50_Bng5dFp16_04_step3", fused_bn_grad_5D_run_3, ((32, 8, 28, 28, 16), "float16", "resnet50_Bng5dFp16_04")),
            # ("resnet50_Bng5dFp16_05_step3", fused_bn_grad_5D_run_3, ((32, 32, 28, 28, 16), "float16", "resnet50_Bng5dFp16_05")),
            # ("resnet50_Bng5dFp16_06_step3", fused_bn_grad_5D_run_3, ((32, 16, 14, 14, 16), "float16", "resnet50_Bng5dFp16_06")),
            # ("resnet50_Bng5dFp16_07_step3", fused_bn_grad_5D_run_3, ((32, 64, 14, 14, 16), "float16", "resnet50_Bng5dFp16_07")),
            # ("resnet50_Bng5dFp16_08_step3", fused_bn_grad_5D_run_3, ((32, 32, 7, 7, 16), "float16", "resnet50_Bng5dFp16_08")),
            # ("resnet50_Bng5dFp16_09_step3", fused_bn_grad_5D_run_3, ((32, 128, 7, 7, 16), "float16", "resnet50_Bng5dFp16_09")),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_aic(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_aic_cloud)

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def test_run_cloud_divided(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.test_5D_split)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == '__main__':
    a = TestCase()
    a.setup()
    a.test_run_cloud()
