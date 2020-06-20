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
bias add test cast
"""

import datetime
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.bias_add_run import bias_add_run


class TestBiasAdd(TestBase):

    def setup(self):
        case_name = "test_akg_bias_add_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            #("bias_add_2d_3_16",       bias_add_run, ([3, 16],         "float16"), [(1, 0), (16, 0)]),
            #("bias_add_2d_5_1024",     bias_add_run, ([5, 1024],       "float16"), [(1, 0), (16, 0)]),
            #("bias_add_3d_3_4_32",     bias_add_run, ([3, 4, 32],      "float16"), [(1, 0), (1, 0), (16, 0)]),
            #("bias_add_4d_23_14_7_16", bias_add_run, ([23, 14, 7, 16], "float16"), [(1, 0), (1, 0), (1, 0), (16, 0)]),
            # ci fail aic ("bias_add_8_2",        bias_add_run, ([8, 2],        "float16"), [(1, 1), (2, 2)]),
            ("bias_add_64_2", bias_add_run, ([64, 2], "DefaultFormat", "float16")),
            ("bias_add_8_1024", bias_add_run, ([8, 1024], "DefaultFormat", "float16")),
            ("bias_add_64_1024", bias_add_run, ([64, 1024], "DefaultFormat", "float16")),
            ("bias_add_160_1024", bias_add_run, ([160, 1024], "DefaultFormat", "float16")),
            ("bias_add_1024_1024", bias_add_run, ([1024, 1024], "DefaultFormat", "float16")),
            ("bias_add_1280_1024", bias_add_run, ([1280, 1024], "DefaultFormat", "float16")),
            #("bias_add_1024_4096",  bias_add_run, ([1024, 4096],  "float16"), [(8, 8), (1024, 1024)]),
            #("bias_add_8192_1024",  bias_add_run, ([8192, 1024],  "float16"), [(8, 8), (1024, 1024)]),
            #("bias_add_160_30522",  bias_add_run, ([160, 30522],  "float16"), [(1, 1), (5087, 5087)]),
            ("bias_add_1_9_9_3", bias_add_run, ([1, 9, 9, 3], "DefaultFormat", "float16")),
            ("bias_add_4_129_129_21", bias_add_run, ([4, 129, 129, 21], "DefaultFormat", "float16")),
            ("bias_add_1_129_129_21", bias_add_run, ([1, 129, 129, 21], "DefaultFormat", "float16")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            #("bias_add_8_2",        bias_add_run, ([8, 2],        "float32"), [(1, 1), (2, 2)]),
        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # ("bias_add_2d_3_16",       bias_add_run, ([3, 16],         "float16"), [(1, 0), (16, 0)]),
            # ("bias_add_2d_5_1024",     bias_add_run, ([5, 1024],       "float16"), [(1, 0), (16, 0)]),
            # ("bias_add_3d_3_4_32",     bias_add_run, ([3, 4, 32],      "float16"), [(1, 0), (1, 0), (16, 0)]),
            # ("bias_add_4d_23_14_7_16", bias_add_run, ([23, 14, 7, 16], "float16"), [(1, 0), (1, 0), (1, 0), (16, 0)]),

            # ("bias_add_8_2",        bias_add_run, ([8, 2],        "float16"), [(1, 1), (2, 2)]),
            # ("bias_add_64_2",       bias_add_run, ([64, 2],       "float16"), [(1, 1), (2, 2)]),
            # ("bias_add_8_1024",     bias_add_run, ([8, 1024],     "float16"), [(8, 8), (1024, 1024)]),
            # ("bias_add_64_1024",    bias_add_run, ([64, 1024],    "float16"), [(8, 8), (1024, 1024)]),
            # ("bias_add_160_1024",   bias_add_run, ([160, 1024],   "float16"), [(8, 8), (1024, 1024)]),
            # ("bias_add_1024_1024",  bias_add_run, ([1024, 1024],  "float16"), [(8, 8), (1024, 1024)]),
            # ("bias_add_1280_1024",  bias_add_run, ([1280, 1024],  "float16"), [(8, 8), (1024, 1024)]),

            # ("bias_add_1024_4096", bias_add_run, ([1024, 4096], "float16"), [(8, 8), (1024, 1024)]),
            # ("bias_add_8192_1024", bias_add_run, ([8192, 1024], "float16"), [(8, 8), (1024, 1024)]),
            # ("bias_add_160_30522", bias_add_run, ([160, 30522], "float16"), [(1, 1), (5087, 5087)]),
        ]
        self.testarg_rpc_cloud = [
            #("bias_add_fp32_002", bias_add_run, ([4,4,2,4], "DefaultFormat", "float32"), [(1, 1), (2, 2), [1,1]]),

            ("bias_add_5d_fp32_001", bias_add_run, ([32, 63, 1, 1, 16], "NC1HWC0", "float32")),
            ("bias_add_fp16_002", bias_add_run, ([32, 1001], "DefaultFormat", "float16"), [(1, 1), (1, 1)]),
            ("bias_add_fp16_003", bias_add_run, ([32, 1, 1, 1001], "NHWC", "float16"), [(1, 1), (1, 1), (1, 1)]),
            ("bias_add_fp32_002", bias_add_run, ([32, 1001], "DefaultFormat", "float32"), [(1, 1), (1, 1)]),
            ("bias_add_fp32_003", bias_add_run, ([32, 1, 1, 1001], "NHWC", "float32"), [(1, 1), (1, 1), (1, 1)]),
        ]

        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
