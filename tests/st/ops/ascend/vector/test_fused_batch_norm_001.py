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

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.fused_batch_norm_run import fused_batch_norm_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_auto_fused_tensor_batch_norm_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # ("Bn2dFp32Ref_01", fused_batch_norm_run, ((64, 512), "float32", 0.9, 1e-5, False, "DefaultFormat", 1, "Bn2dFp32Ref_01")),
            # ("Bn5dFp32Train_02", fused_batch_norm_run, ((32, 4, 7, 7, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "Bn5dFp32Train_02")),
        ]
        self.testarg_aic_cloud = [
            ("Bn5dFp32Train_01", fused_batch_norm_run, ((4, 4, 4, 4, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "Bn5dFp32Train_01")),
        ]
        self.testarg_cloud = [
            #  # caseflag, opfuncname, (shape, dtype, momentum, eps, is_training, data_format, axis, kerne_name)
            #  # 2D
            # ("resnet101_Bn2dFp16Ref01", fused_batch_norm_run, (( 64,  512), "float16", 0.9, 1e-5, False, "DefaultFormat", 1, "resnet101_Bn2dFp16ref01")),
            # ("resnet101_Bn2dFp16Ref02", fused_batch_norm_run, (( 96,  512), "float16", 0.9, 1e-5, False, "DefaultFormat", 1, "resnet101_Bn2dFp16ref02")),
            # ("resnet101_Bn2dFp16Ref03", fused_batch_norm_run, ((128,  512), "float16", 0.9, 1e-5, False, "DefaultFormat", 1, "resnet101_Bn2dFp16ref03")),
            ("resnet101_Bn2dFp32Ref01", fused_batch_norm_run, ((64, 512), "float32", 0.9, 1e-5, False, "DefaultFormat", 1, "resnet101_Bn2dFp32ref01")),
            ("resnet101_Bn2dFp32Ref02", fused_batch_norm_run, ((96, 512), "float32", 0.9, 1e-5, False, "DefaultFormat", 1, "resnet101_Bn2dFp32ref02")),
            ("resnet101_Bn2dFp32Ref03", fused_batch_norm_run, ((128, 512), "float32", 0.9, 1e-5, False, "DefaultFormat", 1, "resnet101_Bn2dFp32ref03")),

            # ("resnet101_Bn2dFp16Train01", fused_batch_norm_run, (( 64,  512), "float16", 0.9, 1e-5, True, "DefaultFormat", 1, "resnet101_Bn2dFp16Train01")),
            # ("resnet101_Bn2dFp16Train02", fused_batch_norm_run, (( 96,  512), "float16", 0.9, 1e-5, True, "DefaultFormat", 1, "resnet101_Bn2dFp16Train02")),
            # ("resnet101_Bn2dFp16Train03", fused_batch_norm_run, ((128,  512), "float16", 0.9, 1e-5, True, "DefaultFormat", 1, "resnet101_Bn2dFp16Train03")),
            ("resnet101_Bn2dFp32Train01", fused_batch_norm_run, ((64, 512), "float32", 0.9, 1e-5, True, "DefaultFormat", 1, "resnet101_Bn2dFp32Train01")),
            ("resnet101_Bn2dFp32Train02", fused_batch_norm_run, ((96, 512), "float32", 0.9, 1e-5, True, "DefaultFormat", 1, "resnet101_Bn2dFp32Train02")),
            ("resnet101_Bn2dFp32Train03", fused_batch_norm_run, ((128, 512), "float32", 0.9, 1e-5, True, "DefaultFormat", 1, "resnet101_Bn2dFp32Train03")),

            #  # 5D
            # ("resnet50_Bn5dFp16Ref01",   fused_batch_norm_run, ((32,   4, 112, 112, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref01")),
            # ("resnet50_Bn5dFp16Ref02",   fused_batch_norm_run, ((32,   4,  56,  56, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref02")),
            # ("resnet50_Bn5dFp16Ref03",   fused_batch_norm_run, ((32,  16, 112, 112, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref03")),
            # ("resnet50_Bn5dFp16Ref04",   fused_batch_norm_run, ((32,   8,  28,  28, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref04")),
            # ("resnet50_Bn5dFp16Ref05",   fused_batch_norm_run, ((32,  32,  28,  28, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref05")),
            # ("resnet50_Bn5dFp16Ref06",   fused_batch_norm_run, ((32,  16,  14,  14, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref06")),
            # ("resnet50_Bn5dFp16Ref07",   fused_batch_norm_run, ((32,  64,  14,  14, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref07")),
            # ("resnet50_Bn5dFp16Ref08",   fused_batch_norm_run, ((32,  32,   7,   7, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref08")),
            # ("resnet50_Bn5dFp16Ref09",   fused_batch_norm_run, ((32, 128,   7,   7, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp16Ref09")),
            # ("resnet101_Bn5dFp16Ref10",  fused_batch_norm_run, ((32,   8,  56,  56, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet101_Bn5dFp16Ref10")),
            # ("resnet101_Bn5dFp16Ref11",  fused_batch_norm_run, ((32,  16,  28,  28, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet101_Bn5dFp16Ref11")),
            # ("resnet101_Bn5dFp16Ref12",  fused_batch_norm_run, ((32,  32,  14,  14, 16), "float16", 0.99, 1e-5, False, "NC1HWC0", None, "resnet101_Bn5dFp16Ref12")),

            # ("resnet50_Bn5dFp16Train01",   fused_batch_norm_run, ((32,   4, 112, 112, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train01")),
            # ("resnet50_Bn5dFp16Train02",   fused_batch_norm_run, ((32,   4,  56,  56, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train02")),
            # ("resnet50_Bn5dFp16Train03",   fused_batch_norm_run, ((32,  16, 112, 112, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train03")),
            # ("resnet50_Bn5dFp16Train04",   fused_batch_norm_run, ((32,   8,  28,  28, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train04")),
            # ("resnet50_Bn5dFp16Train05",   fused_batch_norm_run, ((32,  32,  28,  28, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train05")),
            # ("resnet50_Bn5dFp16Train06",   fused_batch_norm_run, ((32,  16,  14,  14, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train06")),
            # ("resnet50_Bn5dFp16Train07",   fused_batch_norm_run, ((32,  64,  14,  14, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train07")),
            # ("resnet50_Bn5dFp16Train08",   fused_batch_norm_run, ((32,  32,   7,   7, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train08")),
            # ("resnet50_Bn5dFp16Train09",   fused_batch_norm_run, ((32, 128,   7,   7, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Train09")),
            # ("resnet101_Bn5dFp16Train10",  fused_batch_norm_run, ((32,   8,  56,  56, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet101_Bn5dFp16Train10")),
            # ("resnet101_Bn5dFp16Train11",  fused_batch_norm_run, ((32,  16,  28,  28, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet101_Bn5dFp16Train11")),
            # ("resnet101_Bn5dFp16Train12",  fused_batch_norm_run, ((32,  32,  14,  14, 16), "float16", 0.99, 1e-5, True, "NC1HWC0", None, "resnet101_Bn5dFp16Train12")),

            # ("resnet50_Bn5dFp32Ref01",   fused_batch_norm_run, ((32,   4, 112, 112, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref01")),
            # ("resnet50_Bn5dFp32Ref02",   fused_batch_norm_run, ((32,   4,  56,  56, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref02")),
            # ("resnet50_Bn5dFp32Ref03",   fused_batch_norm_run, ((32,  16, 112, 112, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref03")),
            ("resnet50_Bn5dFp32Ref04", fused_batch_norm_run, ((32, 8, 28, 28, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref04")),
            # ("resnet50_Bn5dFp32Ref05",   fused_batch_norm_run, ((32,  32,  28,  28, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref05")),
            ("resnet50_Bn5dFp32Ref06", fused_batch_norm_run, ((32, 16, 14, 14, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref06")),
            # ("resnet50_Bn5dFp32Ref07",   fused_batch_norm_run, ((32,  64,  14,  14, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref07")),
            # ("resnet50_Bn5dFp32Ref08",   fused_batch_norm_run, ((32,  32,   7,   7, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref08")),
            # ("resnet50_Bn5dFp32Ref09",   fused_batch_norm_run, ((32, 128,   7,   7, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet50_Bn5dFp32Ref09")),
            ("resnet101_Bn5dFp32Ref10", fused_batch_norm_run, ((32, 8, 56, 56, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet101_Bn5dFp32Ref10")),
            # ("resnet101_Bn5dFp32Ref11",  fused_batch_norm_run, ((32,  16,  28,  28, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet101_Bn5dFp32Ref11")),
            # ("resnet101_Bn5dFp32Ref12",  fused_batch_norm_run, ((32,  32,  14,  14, 16), "float32", 0.99, 1e-5, False, "NC1HWC0", None, "resnet101_Bn5dFp32Ref12")),

            # resnet50 V1.0 shapes
            ("resnet50_Bn5dFp16Ref01", fused_batch_norm_run, ((32, 4, 112, 112, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref01")),
            ("resnet50_Bn5dFp16Ref02", fused_batch_norm_run, ((32, 4, 56, 56, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref02")),
            ("resnet50_Bn5dFp16Ref03", fused_batch_norm_run, ((32, 16, 56, 56, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref03")),
            ("resnet50_Bn5dFp16Ref04", fused_batch_norm_run, ((32, 8, 28, 28, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref04")),
            ("resnet50_Bn5dFp16Ref05", fused_batch_norm_run, ((32, 32, 28, 28, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref05")),
            ("resnet50_Bn5dFp16Ref06", fused_batch_norm_run, ((32, 16, 14, 14, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref06")),
            ("resnet50_Bn5dFp16Ref07", fused_batch_norm_run, ((32, 64, 14, 14, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref07")),
            ("resnet50_Bn5dFp16Ref08", fused_batch_norm_run, ((32, 32, 7, 7, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref08")),
            ("resnet50_Bn5dFp16Ref09", fused_batch_norm_run, ((32, 128, 7, 7, 16), "float32", 0.99, 1e-5, True, "NC1HWC0", None, "resnet50_Bn5dFp16Ref09")),
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

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
