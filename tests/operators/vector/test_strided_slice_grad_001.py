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
from base import TestBase
import pytest


class TestCase(TestBase):
    def setup(self):
        case_name = "test_auto_strided_slice_grad_run_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("strided_slice_grad_dim1", "strided_slice_grad_run",
             [(100,), [10], [50], [3], 0, 0, 0, 0, 0, (14,), "float16"]),
            ("strided_slice_grad_dim2_1", "strided_slice_grad_run",
             [(2, 2), [0, 0], [2, 2], [1, 1], 0, 0, 0, 0, 0, (2, 2), "float16"]),
            ("strided_slice_grad_dim2_2", "strided_slice_grad_run",
             [(3, 5), [0, 0, 0], [0, 3, 5], [1, 2, 3], 0, 0, 0, 1, 0, (1, 2, 2), "float16"]),
            ("strided_slice_grad_dim2_3", "strided_slice_grad_run",
             [(23, 26), [7, 10], [20, 21], [2, 3], 0, 0, 0, 0, 0, (7, 4), "float16"]),
            ("strided_slice_grad_dim3_1", "strided_slice_grad_run",
             [[6, 16, 26], [0, 0, 0], [6, 1, 26], [1, 1, 1], 0, 0, 0, 0, 0, (6, 1, 26), "float16"]),
            ("strided_slice_grad_dim3_2", "strided_slice_grad_run",
             [(5, 5, 5), [1, 1, 1], [4, 4, 4], [2, 2, 2], 0, 0, 0, 0, 0, (2, 2, 2), "float16"]),
            ("strided_slice_grad_dim4_1", "strided_slice_grad_run",
             [(5, 5, 5, 5), [1, 1, 1, 1], [4, 4, 4, 4], [2, 2, 2, 2], 0, 0, 0, 0, 0, (2, 2, 2, 2), "float16"]),
            ("strided_slice_grad_dim4_2", "strided_slice_grad_run",
             [(16, 16, 16, 16), [0, 1, 1, 0], [0, 2, 0, 16], [1, 1, 1, 2], 9, 5, 0, 0, 2, (16, 15, 8), "float16"]),

            ("strided_slice_grad_dim5", "strided_slice_grad_run",
             [(5, 5, 5, 5, 5), [1, 1, 1, 1, 1], [4, 4, 4, 4, 4], [2, 2, 2, 2, 2], 0, 0, 0, 0, 0, (2, 2, 2, 2, 2),
              "float16"]),
            ("strided_slice_grad_dim6", "strided_slice_grad_run",
             [(5, 5, 5, 5, 5, 5), [1, 1, 1, 1, 1, 1], [4, 4, 4, 4, 4, 4], [2, 2, 2, 2, 2, 2], 0, 0, 0, 0, 0,
              (2, 2, 2, 2, 2, 2), "float16"]),

            # grad shape is illegal.
            # ("strided_slice_grad_dim7_1", "strided_slice_grad_run",
            #  [(1, 5, 5, 5, 5, 5, 5), [1, 1, 1, 1, 1, 1, 1], [2, 4, 4, 4, 4, 4, 4], [1, 2, 2, 2, 2, 2, 2], 0, 0, 0, 0, 0,
            #   (1, 2, 2, 2, 2, 2, 2), "float16"]),

            # Data alignment is illegal.
            # ("strided_slice_grad_dim7_2", "strided_slice_grad_run", [(5, 5, 5, 5, 5, 5, 5), [1, 1, 1, 1, 1, 1, 1], [5, 4, 4, 4, 4, 4, 4], [3, 2, 2, 2, 2, 2, 2], 0, 0, 0, 0, 0, (2, 2, 2, 2, 2, 2, 2), "float16"]),

            ("strided_slice_grad_dim8_1", "strided_slice_grad_run",
             [(1, 1, 5, 5, 5, 5, 5, 5), [0, 0, 0, 0, 0, 0, 0, 0], [1, 1, 5, 5, 5, 5, 5, 5], [1, 1, 2, 2, 2, 2, 2, 2], 0,
              0, 0, 0, 0, (1, 1, 3, 3, 3, 3, 3, 3), "float16"]),
            ("strided_slice_grad_dim8_2", "strided_slice_grad_run", (
                (5, 2, 3, 4, 5, 6, 16), [0, 1, 1, 0, 0, 0, 0], [0, 2, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1], 17, 5, 32, 72,
                2, (5, 2, 1, 2, 5, 6, 16, 1), "float16")),
            ("strided_slice_grad_dim3", "strided_slice_grad_run",
             [[64, 512, 1024], [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (64, 1, 1024), "float16"]),
        ]

        self.testarg_rpc_cloud = [
            ("strided_slice_grad_dim3_float32", "strided_slice_grad_run",
             [(64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, (64, 1, 1024), "float32"]),
            ("strided_slice_grad_dim3_int32_004", "strided_slice_grad_run",
             [(1, 1, 768), [0, 0, 0], [1, 128, 768], [1, 1, 1], 0, 0, 0, 0, 0, (1, 1, 768), "int32"]),
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

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run([self.testarg_rpc_cloud[0]])

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
