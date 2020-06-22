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
matmul
"""
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_matmul_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            # shape_x, shape_y, bias, left_format, right_format, output_format, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs

            # bert shape
            ("matmul_run_bert_00", "matmul_run", ((16, 1024), (16, 1024), 0, "zN", "zN", "zN", False, True, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_01", "matmul_run", ((8192, 4096), (8192, 1024), 0, "zN", "zN", "zN", True, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_02", "matmul_run", ((8192, 1024), (1024, 4096), 0, "zN", "zN", "zN", False, False, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_03", "matmul_run", ((16, 16), (16, 1024), 0, "zN", "zN", "zN", True, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_04", "matmul_run", ((1216, 1024), (1024, 1024), 0, "zN", "zN", "zN", False, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_05", "matmul_run", ((8192, 4096), (4096, 1024), 0, "zN", "zN", "zN", False, False, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_06", "matmul_run", ((8192, 1024), (4096, 1024), 0, "zN", "zN", "zN", False, True, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_07", "matmul_run", ((8192, 1024), (8192, 4096), 0, "zN", "zN", "zN", True, False, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_08", "matmul_run", ((1216, 1024), (1024, 1024), 0, "zN", "zN", "zN", False, True, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_09", "matmul_run", ((8192, 1024), (1024, 1024), 0, "zN", "zN", "zN", False, False, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_10", "matmul_run", ((1216, 30522), (30522, 1024), 0, "zN", "zN", "zN", False, False, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_11", "matmul_run", ((1216, 30522), (1216, 1024), 0, "zN", "zN", "zN", True, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_12", "matmul_run", ((1216, 1024), (30522, 1024), 0, "zN", "zN", "zN", False, True, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_13", "matmul_run", ((8192, 1024), (8192, 1024), 0, "zN", "zN", "zN", True, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_14", "matmul_run", ((1216, 1024), (1216, 1024), 0, "zN", "zN", "zN", True, False, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_15", "matmul_run", ((16, 1024), (16, 1024), 0, "zN", "zN", "zN", True, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_16", "matmul_run", ((16, 1024), (1024, 1024), 0, "zN", "zN", "zN", False, True, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_17", "matmul_run", ((16, 16), (16, 1024), 0, "zN", "zN", "zN", False, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_bert_18", "matmul_run", ((8192, 1024), (1024, 1024), 0, "zN", "zN", "zN", False, True, "float16", "float16", "matmul_cce")),
            ("matmul_run_bert_19", "matmul_run", ((8192, 4096), (1024, 4096), 0, "zN", "zN", "zN", False, True, "float16", "float16", "matmul_cce")),

            # matmul_cast
            ("matmul_run1", "matmul_run",
             ((64, 1024), (16, 1024), 0, "zZ", "nZ", "zN", False, True, "float16", "float32", "matmul_cast_cce")),
            # ((4, 4), (16, 16), (128, 128), (16, 16), (16, 16))),
            # matmul_bias
            ("matmul_run2", "matmul_run",
             ((64, 1024), (16, 1024), 1, "zZ", "nZ", "zN", False, True, "float16", "float16", "matmul_bias_cce")),
            # ((4, 4), (16, 16), (128, 128), (16, 16), (16, 16))),
            # matmul_trans
            ("matmul_run3", "matmul_run",
             ((1024, 64), (16, 1024), 1, "zZ", "nZ", "zN", True, True, "float16", "float16", "matmul_bias_cce")),
            # ((4, 4), (16, 16), (128, 128), (16, 16), (16, 16))),

            # matmul
            ("matmul_run4", "matmul_run",
             ((64, 1024), (16, 1024), 0, "zZ", "nZ", "zN", False, True, "float16", "float16", "matmul_cce")),
            # ((4, 4), (16, 16), (128, 128), (16, 16), (16, 16))),
            ("matmul_run5", "matmul_run",
             ((1024, 16), (16, 1024), 1, "zZ", "nZ", "zN", False, False, "float16", "float16", "matmul_cce")),
            # ((8, 8), (8, 8), (128, 128), (128, 128), (16, 16))),
            ("matmul_run9", "matmul_run",
             ((16, 1024), (16, 1024), 0, "zZ", "nZ", "zN", False, True, "float16", "float16", "matmul_cce")),
            # ((16, 16), (16, 16), (16, 16))),
            ("matmul_run16", "matmul_run",
             ((16, 64), (64, 1024), 0, "zZ", "nZ", "zN", False, False, "float16", "float16", "matmul_cce")),
            # ((16, 16), (16, 16), (16, 16), (4, 4))),

            # new shape for bert
            # ("matmul_run29", "matmul_run",
            # ((8192,2), (1024,2), 0, 0, False, True,  "float16", "float16", "matmul_cce"),
            # ((8, 8), (8, 8), (128, 128), (128, 128), (16, 16))),

            ("matmul_run30", "matmul_run",
             ((64, 1024), (2, 1024), 0, "zZ", "nZ", "zN", False, True, "float16", "float16", "matmul_cce")),
            # ((4, 4), (16, 16), (16, 16), (16, 16), (16, 16))),

            ("matmul_run31", "matmul_run",
             ((2, 64), (1024, 64), 0, "zZ", "nZ", "zN", False, True, "float16", "float16", "matmul_cce")),
            # ((16, 16), (16, 16), (16, 16), (16, 16))),

            # zZ case
            ("matmul_run1", "matmul_run",
             ((6272, 256), (6272, 256), 0, "zZ", "zZ", "zZ", True, False, "float16", "float32", "matmul_cast_cce")),
            ("matmul_run2", "matmul_run",
             ((6272*16, 4*16), (6272*16, 4*16), 0, "zZ", "zZ", "zZ", True, False, "float16", "float32", "matmul_cce")),
            ("matmul_run3", "matmul_run",
             ((1568*16, 8*16), (1568*16, 8*16), 0, "zZ", "zZ", "zZ", True, False, "float16", "float32", "matmul_cce")),

            # zN case
            ("matmul_run_zN_1", "matmul_run",
             ((32, 48), (48, 64), 0, "zN", "zN", "zN", False, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_zN_2", "matmul_run",
             ((32, 48), (48, 64), 0, "zN", "zN", "zN", True, False, "float16", "float32", "matmul_cce")),
            ("matmul_run_zN_3", "matmul_run",
             ((32, 48), (48, 64), 0, "zN", "zN", "zN", False, True, "float16", "float32", "matmul_cce")),
        ]

        self.testarg_rpc_cloud = [
            # # float - float:[160, 1024] - [1024, 1024] = float:[160, 1024]

        ]
        self.testarg_level1 = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            #shape_x, shape_y, bias, left_format, right_format, output_format, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs

            ("matmul_run29", "matmul_run",
             ((8192, 16), (1024, 16), 0, "zZ", "nZ", "zN", False, True, "float16", "float16", "matmul_cce"),
             ((8, 8), (8, 8), (128, 128), (128, 128), (128, 128))),

            # ("matmul_run33", "matmul_run",
            #  ((16, 32), (32, 32), 0, 0, False, True, "float16", "float16", "matmul_cce"),
            #  ((4, 8), (4,8), (16, 128), (16, 128), (16, 128))),
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
    def test_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run([self.testarg_rpc_cloud[0]])

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
