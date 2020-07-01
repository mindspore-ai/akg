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
batch_norm
"""
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_batchmatmul_run_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # bs, m, n, k, bias_shape, dtype, kernel_name, attrs
            ("batch_matmul_001", "batchmatmul_run",
             ((4,), 16, 48, 32, (1,), "float32", False, True, "batch_matmul_output")),
            #("batch_matmul_002", "batchmatmul_run",
            # ((4,), 16, 48, 32, (48,), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_003", "batchmatmul_run",
             ((4,), 16, 48, 32, (4, 16, 48), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_004", "batchmatmul_run", ((), 16, 48, 32, (), "float32", True, False, "batch_matmul_output")),
            ("batch_matmul_005", "batchmatmul_run", ((), 16, 48, 32, (), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_006", "batchmatmul_run",
             ((4, 2), 16, 48, 32, (1, 1), "float32", False, False, "batch_matmul_output")),
            #("batch_matmul_007", "batchmatmul_run",
            # ((4, 2), 16, 48, 32, (1, 48), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_008", "batchmatmul_run",
             ((4, 2), 16, 48, 32, (4, 2, 16, 48), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_009", "batchmatmul_run",
             ((4, 2), 16, 48, 32, (), "float32", True, False, "batch_matmul_output")),
            ("batch_matmul_010", "batchmatmul_run",
             ((8, 16), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            ############################
            # for bert small case add by mini in ci
            ############################
            # ("matmul_0033", "batchmatmul_run", ((), 3072, 768, 8192, (), "float16", True, False, "batch_matmul_bert")),
            # ("matmul_0037", "batchmatmul_run", ((), 33, 64, 16384, (), "float32", True, False, "batch_matmul_bert")),
            ("matmul_0053", "batchmatmul_run", ((), 32000, 768, 20, (), "float32", True, False, "batch_matmul_bert")),
            ('matmul_0060', "batchmatmul_run", ((), 20, 768, 32000, (), 'float32', False, False, 'batchmatmul_bert')),
            ('matmul_0061', "batchmatmul_run", ((128,), 768, 64, 128, (), 'float32', False, False, 'batchmatmul_bert')),
            # ('matmul_0062', "batchmatmul_run", ((), 16384, 6384, 33, (), 'float32', True, False, 'batchmatmul_bert')),
            ('matmul_0063', "batchmatmul_run", ((), 32000, 768, 20, (), 'float32', False, False, 'batchmatmul_bert')),
        ]
        self.testarg_cloud = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # bs, m, n, k, bias_shape, dtype, kernel_name, attrs
            (
                "batch_matmul_001", "batchmatmul_run", ((), 16, 48, 32, (1,), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_002", "batchmatmul_run",
             ((), 16, 48, 32, (48,), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_003", "batchmatmul_run",
             ((), 16, 48, 32, (1, 1), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_004", "batchmatmul_run",
             ((), 16, 48, 32, (16, 1), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_005", "batchmatmul_run",
             ((), 16, 48, 32, (1, 48), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_006", "batchmatmul_run",
             ((), 16, 48, 32, (16, 48), "float32", False, True, "batch_matmul_output")),
            # ("batch_matmul_007", "batchmatmul_run", ((64, 12), 128, 128, 64, (), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_001", "batchmatmul_run",
             ((1,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
        ]

        self.testarg_rpc_cloud = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # bs, m, n, k, bias_shape, dtype, kernel_name, attrs
            # 4D
            # ("batch_matmul_4D_001", "batchmatmul_run", ((128,), 128, 64, 768, (), "float32", True, False, "batch_matmul_output")),
            # ("batch_matmul_4D_002", "batchmatmul_run", ((64, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # ("batch_matmul_4D_003", "batchmatmul_run", ((128,), 768, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # ("batch_matmul_4D_004", "batchmatmul_run", ((64, 12), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # ("batch_matmul_4D_005", "batchmatmul_run", ((128,), 768, 64, 128, (), "float32", False, False, "batch_matmul_output")),

            # caseflag, opfuncname, testRunArgs, dimArgs
            # bs, m, n, k, bias_shape, dtype, kernel_name, attrs
            ("batch_matmul_007", "batchmatmul_run",
             ((64, 12), 128, 128, 64, (), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_007", "batchmatmul_run",
             ((1, 12), 128, 128, 64, (), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_001", "batchmatmul_run",
             ((1, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),

            # Matrix (2D)
            ("batch_matmul_2D_001", "batchmatmul_run",
             ((), 8192, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_2D_007", "batchmatmul_run",
             ((), 64, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_2D_008", "batchmatmul_run",
             ((), 8192, 768, 21128, (), "float32", False, False, "batch_matmul_output"),
             ((16, 16), (16, 16), (16, 16))),
            ("batch_matmul_2D_009", "batchmatmul_run",
             ((), 8192, 768, 3072, (), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_2D_012", "batchmatmul_run",
             ((), 64, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_2D_013", "batchmatmul_run",
             ((), 8192, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_2D_014", "batchmatmul_run",
             ((), 8192, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_2D_015", "batchmatmul_run",
             ((), 64, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            ("batch_matmul_2D_016", "batchmatmul_run",
             ((), 21128, 768, 8192, (), "float32", False, True, "batch_matmul_output")),
            ("batch_matmul_2D_017", "batchmatmul_run",
             ((), 768, 768, 1280, (), "float32", True, False, "batch_matmul_output")),

            # # float - float:[64, 16, 128, 64] - [64, 16, 128, 64] = float:[64, 16, 128, 128]
            ("batch_matmul_4D_001", "batchmatmul_run",
             ((64, 16), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[8, 16, 128, 64] - [8, 16, 128, 64] = float:[8, 16, 128, 128]
            ("batch_matmul_4D_002", "batchmatmul_run",
             ((8, 16), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[64, 16, 128, 128] - [64, 16, 128, 64] = float:[64, 16, 128, 64]
            ("batch_matmul_4D_003", "batchmatmul_run",
             ((64, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 16, 128, 128] - [8, 16, 128, 64] = float:[8, 16, 128, 64]
            ("batch_matmul_4D_004", "batchmatmul_run",
             ((8, 16), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),

            # half - half:[128, 768, 128] - [128, 768, 64] = half:[128, 128, 64]
            ("batch_matmul_3D_005", "batchmatmul_run",
             ((128,), 128, 64, 768, (), "float16", True, False, "batch_matmul_output")),
            # half - half:[64, 12, 128, 128] - [64, 12, 128, 64] = half:[64, 12, 128, 64]
            ("batch_matmul_4D_006", "batchmatmul_run",
             ((64, 12), 128, 64, 128, (), "float16", False, False, "batch_matmul_output")),

            # # half - half:[128, 768, 64] - [128, 128, 64] = half:[128, 768, 128]
            ("batch_matmul_3D_007", "batchmatmul_run",
             ((128,), 768, 128, 64, (), "float16", False, True, "batch_matmul_output")),
            # # half - half:[64, 12, 128, 64] - [64, 12, 128, 64] = half:[64, 12, 128, 128]
            ("batch_matmul_4D_008", "batchmatmul_run",
             ((64, 12), 128, 128, 64, (), "float16", False, True, "batch_matmul_output")),
            # # half - half:[128, 768, 128] - [128, 128, 64] = half:[128, 768, 64]
            ("batch_matmul_3D_009", "batchmatmul_run",
             ((128,), 768, 64, 128, (), "float16", False, False, "batch_matmul_output")),
            #  cost a long time
            #  3461 seconds for below this by run on 1980
            # ("batch_matmul_2D_17", "batchmatmul_run", ((), 30522, 1024, 1280, False, "float32", True, False, "batch_matmul_output"), ((32, 32), (32, 32), (32, 32))),
            #  3569 seconds for below this by run on 1980
            # ("batch_matmul_2D_29", "batchmatmul_run", ((), 1280, 1024, 30522, False, "float32", False, False, "batch_matmul_output"), ((32, 32), (32, 32), (32, 32))),

            #  fail for now,
            #  As do not support that trans_a and trans_b both true:
            # ("batch_matmul_2D_27", "batchmatmul_run", ((), 1024, 1024, 64, False, "float32", True, True, "batch_matmul_output")),

            # half - half:[8192, 3072] - [768, 3072] = half:[8192, 768]
            ("matmul_0043", "batchmatmul_run",
             ((), 8192, 768, 3072, (), "float16", False, True, "batch_matmul_output_fp16")),
            # half - half:[8192, 768] - [3072, 768] = half:[8192, 3072]
            ("matmul_0044", "batchmatmul_run",
             ((), 8192, 3072, 768, (), "float16", False, True, "batch_matmul_output_fp16")),
            # half - half:[8192, 768] - [768, 768] = half:[8192, 768]
            ("matmul_0048", "batchmatmul_run",
             ((), 8192, 768, 768, (), "float16", False, False, "batch_matmul_output_fp16")),

            #  error: Not all Vars are passed in api_args:  'cc5'  'cc5'  'cc5'  does not appear in api_args
            # ("matmul_0029", "batchmatmul_run", ((), 768, 768, 8192, (), "float16", True, False, "batch_matmul_output"), ((1,1),(16,1),(1024,1))),
            # ("matmul_0033", "batchmatmul_run", ((), 3072, 768, 8192, (), "float16", True, False, "batch_matmul_output"), ((1,1),(16,1),(1024,1))),

            ("matmul_0036", "batchmatmul_run",
             ((), 768, 3072, 768, (), "float16", False, False, "batch_matmul_output_fp16")),
            # half - half:[8192, 768] - [768, 3072] = half:[8192, 3072]
            ("matmul_0035", "batchmatmul_run",
             ((), 8192, 3072, 768, (), "float16", False, False, "batch_matmul_output_fp16")),
            # # half - half:[8192, 3072] - [3072, 768] = half:[8192, 768]
            ("matmul_0052", "batchmatmul_run",
             ((), 8192, 768, 3072, (), "float16", False, False, "batch_matmul_output_fp16")),

            # lenet
            ('matmul_lenet_001_fp32', "batchmatmul_run", ((), 1, 120, 784, (120,), 'float32', False, True, 'batchmatmul_output')),
            ('matmul_lenet_002_fp32', "batchmatmul_run", ((), 1, 84, 120, (84,), 'float32', False, True, 'batchmatmul_output')),
            ('matmul_lenet_003_fp32', "batchmatmul_run", ((), 1, 10, 84, (10,), 'float32', False, True, 'batchmatmul_output')),
            ('matmul_lenet_004_fp32', "batchmatmul_run", ((), 10, 84, 1, (), 'float32', True, False, 'batchmatmul_output')),
            ('matmul_lenet_005_fp32', "batchmatmul_run", ((), 1, 84, 10, (), 'float32', False, False, 'batchmatmul_output')),
            ('matmul_lenet_006_fp32', "batchmatmul_run", ((), 84, 120, 1, (), 'float32', True, False, 'batchmatmul_output')),
            ('matmul_lenet_007_fp32', "batchmatmul_run", ((), 1, 120, 84, (), 'float32', False, False, 'batchmatmul_output')),
            ('matmul_lenet_008_fp16', "batchmatmul_run", ((), 120, 784, 1, (), 'float16', True, False, 'batchmatmul_output')),
            ('matmul_lenet_009_fp16', "batchmatmul_run", ((), 1, 784, 120, (), 'float32', False, False, 'batchmatmul_output')),
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

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run([self.testarg_rpc_cloud[0]])

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
