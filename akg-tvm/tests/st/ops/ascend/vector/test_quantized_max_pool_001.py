# Copyright 2020 Huawei Technologies Co., Ltd
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

"""quantized_max_pool test case"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.quantized_max_pool_run import quantized_max_pool_run


class TestQuantizedMaxPool(TestBase):
    """test case class for quantized_max_pool"""
    def setup(self):
        case_name = "test_akg_quantized_max_pool_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """setup case parameters for test"""
        self.caseresult = True
        self._log.info("=================%s Setup case=================", self.casename)
        self.testarg_mini = [
            # testflag, opfunc, (shape, dtype1, shape_list, dtype2,
            #                    ksize, strides, padding, data_format,
            #                    quant_algo, scale_mode, scale_sqrt), dimArgs
            ("qmaxpool_mini_01", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 4, 4, 1), (1, 1, 3, 3, 1), "VALID", "NC1HWC0",
                [1, 0], 2, 0)),
            ("qmaxpool_mini_02", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 4, 4), (1, 1, 3, 3), "VALID", "NCHW", [1, 0], 2, 0)),
            ("qmaxpool_mini_03", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", ((1,), (1,)), "float16",
                (1, 4, 4, 1), (1, 3, 3, 1), "VALID", "NHWC", [1, 0], 2, 0)),
            ("qmaxpool_mini_04", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", None, None,
                (1, 1, 4, 4, 1), (1, 1, 3, 3, 1), "VALID", "NC1HWC0",
                None, None, None)),
            ("qmaxpool_mini_05", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 4, 4, 1), (1, 1, 3, 3, 1), "VALID", "NC1HWC0",
                [0, 0], 2, 0)),
            ("qmaxpool_mini_06", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 4, 4, 1), (1, 1, 3, 3, 1), "SAME", "NC1HWC0",
                [1, 0], 2, 0)),
            ("qmaxpool_mini_07", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 4, 4), (1, 1, 3, 3), "SAME", "NCHW", [1, 0], 2, 0)),
            ("qmaxpool_mini_08", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", ((1,), (1,)), "float16",
                (1, 4, 4, 1), (1, 3, 3, 1), "SAME", "NHWC", [1, 0], 2, 0)),
            ("qmaxpool_mini_09", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", None, None,
                (1, 1, 4, 4, 1), (1, 1, 3, 3, 1), "SAME", "NC1HWC0",
                None, None, None)),
            ("qmaxpool_mini_10", quantized_max_pool_run, (
                (1, 1, 16, 16, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 4, 4, 1), (1, 1, 3, 3, 1), "SAME", "NC1HWC0",
                [0, 0], 2, 0)),
        ]
        self.testarg_cloud = [
            ("qmaxpool_mini_05", quantized_max_pool_run, (
                (1, 1, 64, 64, 16), "float16", None, None,
                (1, 1, 4, 4, 1), (1, 1, 3, 3, 1), "VALID", "NC1HWC0",
                None, None, None)),
            ("qmaxpool_mini_05", quantized_max_pool_run, (
                (1, 1, 64, 64, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 4, 4, 1), (1, 1, 3, 3, 1), "VALID", "NC1HWC0",
                [0, 0], 2, 0)),
            ("qmaxpool_cld_big", quantized_max_pool_run, (
                (32, 4, 112, 112, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 3, 3, 1), (1, 1, 2, 2, 1), "SAME", "NC1HWC0",
                [0, 0], 2, 0)),
            ("qmaxpool_cld_big", quantized_max_pool_run, (
                (32, 4, 112, 112, 16), "float16", ((1,), (1,)), "float16",
                (1, 1, 3, 3, 1), (1, 1, 2, 2, 1), "SAME", "NC1HWC0",
                [1, 0], 2, 0)),
        ]

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_mini_run(self):
        """run case for mini"""
        self.common_run(self.testarg_mini[0:3])

    def test_cloud_run(self):
        """run case for cloud"""
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """clean environment"""
        self._log.info("=============%s Teardown===========", self.casename)
