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
from tests.common.base import TestBase


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_batchmatmul_run_002"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # bs, m, n, k, bias_shape, dtype, kernel_name, attrs
        ]
        self.testarg_cloud = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # bs, m, n, k, bias_shape, dtype, kernel_name, attrs
        ]

        self.testarg_rpc_cloud = [
            # this shape is for new bert shape
            # caseflag, opfuncname, testRunArgs, dimArgs
            # case_name, "batchmatmul_run", bs, m, n, k, bias_shape, dtype, trans_a, trans_b, kernel_name, attrs

            # float-float:[1024, 768] - [768, 3072] = float:[1024, 3072]
            ("batch_matmul_2D_001", "batchmatmul_run", ((), 1024, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1024, 3072] - [3072, 768] = float:[1024, 768]
            ("batch_matmul_2D_002", "batchmatmul_run", ((), 1024, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1024, 768] - [768, 768] = float:[1024, 768]
            ("batch_matmul_2D_003", "batchmatmul_run", ((), 1024, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1, 128, 1] - [1, 1, 128] = float:[1, 128, 128]
            ("batch_matmul_3D_004", "batchmatmul_run", ((1,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 12, 128, 64] - [128, 12, 128, 64] = float:[128, 12, 128, 128]
            ("batch_matmul_4D_005", "batchmatmul_run", ((128, 12), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 12, 128, 128] - [128, 12, 128, 64] = float:[128, 12, 128, 64]
            ("batch_matmul_4D_006", "batchmatmul_run", ((128, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 12, 64] - [128, 128, 64] = float:[128, 12, 128]
            ("batch_matmul_3D_007", "batchmatmul_run", ((128,), 12, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 12, 128] - [128, 128, 64] = float:[128, 12, 64]
            ("batch_matmul_3D_008", "batchmatmul_run", ((128,), 12, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 128, 1] - [128, 1, 128] = float:[128, 128, 128]
            ("batch_matmul_3D_009", "batchmatmul_run", ((128,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 12, 128] - [128, 12, 64] = float:[128, 128, 64]
            ("batch_matmul_3D_010", "batchmatmul_run", ((128,), 128, 64, 12, (), "float32", True, False, "batch_matmul_output")),

            # float - float:[128, 1536, 128] - [128, 1536, 64] = float:[128, 128, 64]
            ("batch_matmul_3D_011", "batchmatmul_run", ((128,), 128, 64, 1536, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 192, 128] - [128, 192, 64] = float:[128, 128, 64]
            ("batch_matmul_3D_012", "batchmatmul_run", ((128,), 128, 64, 192, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 24, 128] - [128, 24, 64] = float:[128, 128, 64]
            ("batch_matmul_3D_013", "batchmatmul_run", ((128,), 128, 64, 24, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 384, 128] - [128, 384, 64] = float:[128, 128, 64]
            ("batch_matmul_3D_014", "batchmatmul_run", ((128,), 128, 64, 384, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 48, 128] - [128, 48, 64] = float:[128, 128, 64]
            ("batch_matmul_3D_015", "batchmatmul_run", ((128,), 128, 64, 48, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 768, 128] - [128, 768, 64] = float:[128, 128, 64]
            ("batch_matmul_3D_016", "batchmatmul_run", ((128,), 128, 64, 768, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 96, 128] - [128, 96, 64] = float:[128, 128, 64]
            ("batch_matmul_3D_017", "batchmatmul_run", ((128,), 128, 64, 96, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[128, 1536, 64] - [128, 128, 64] = float:[128, 1536, 128]
            ("batch_matmul_3D_018", "batchmatmul_run", ((128,), 1536, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 1536, 128] - [128, 128, 64] = float:[128, 1536, 64]
            ("batch_matmul_3D_019", "batchmatmul_run", ((128,), 1536, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 192, 64] - [128, 128, 64] = float:[128, 192, 128]
            ("batch_matmul_3D_020", "batchmatmul_run", ((128,), 192, 128, 64, (), "float32", False, True, "batch_matmul_output")),

            # float - float:[128, 192, 128] - [128, 128, 64] = float:[128, 192, 64]
            ("batch_matmul_3D_021", "batchmatmul_run", ((128,), 192, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 24, 64] - [128, 128, 64] = float:[128, 24, 128]
            ("batch_matmul_3D_022", "batchmatmul_run", ((128,), 24, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 24, 128] - [128, 128, 64] = float:[128, 24, 64]
            ("batch_matmul_3D_023", "batchmatmul_run", ((128,), 24, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 768] - [2, 768] = float:[128, 2]
            ("batch_matmul_2D_024", "batchmatmul_run", ((), 128, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 384, 64] - [128, 128, 64] = float:[128, 384, 128]
            ("batch_matmul_3D_025", "batchmatmul_run", ((128,), 384, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 384, 128] - [128, 128, 64] = float:[128, 384, 64]
            ("batch_matmul_3D_026", "batchmatmul_run", ((128,), 384, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 48, 64] - [128, 128, 64] = float:[128, 48, 128]
            ("batch_matmul_3D_027", "batchmatmul_run", ((128,), 48, 128, 64, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 48, 128] - [128, 128, 64] = float:[128, 48, 64]
            ("batch_matmul_3D_028", "batchmatmul_run", ((128,), 48, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 768, 64] - [128, 128, 64] = float:[128, 768, 128]
            ("batch_matmul_3D_029", "batchmatmul_run", ((128,), 768, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 768, 128] - [128, 128, 64] = float:[128, 768, 64]
            ("batch_matmul_3D_030", "batchmatmul_run", ((128,), 768, 64, 128, (), "float32", False, False, "batch_matmul_output")),

            # float - float:[128, 2] - [2, 768] = float:[128, 768]
            ("batch_matmul_2D_031", "batchmatmul_run", ((), 128, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 3072] - [3072, 768] = float:[128, 768]
            ("batch_matmul_2D_032", "batchmatmul_run", ((), 128, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 768] - [768, 768] = float:[128, 768]
            ("batch_matmul_2D_033", "batchmatmul_run", ((), 128, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 96, 64] - [128, 128, 64] = float:[128, 96, 128]
            ("batch_matmul_3D_034", "batchmatmul_run", ((128,), 96, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[128, 96, 128] - [128, 128, 64] = float:[128, 96, 64]
            ("batch_matmul_3D_035", "batchmatmul_run", ((128,), 96, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[160, 768] - [21128, 768] = float:[160, 21128]
            ("batch_matmul_2D_036", "batchmatmul_run", ((), 160, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[160, 21128] - [21128, 768] = float:[160, 768]
            ("batch_matmul_2D_037", "batchmatmul_run", ((), 160, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[160, 768] - [768, 768] = float:[160, 768]
            ("batch_matmul_2D_038", "batchmatmul_run", ((), 160, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 12, 128, 64] - [16, 12, 128, 64] = float:[16, 12, 128, 128]
            ("batch_matmul_4D_039", "batchmatmul_run", ((16, 12), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[16, 12, 128, 128] - [16, 12, 128, 64] = float:[16, 12, 128, 64]
            ("batch_matmul_4D_040", "batchmatmul_run", ((16, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),

            # float - float:[16, 128, 1] - [16, 1, 128] = float:[16, 128, 128]
            ("batch_matmul_3D_041", "batchmatmul_run", ((16, ), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 768] - [2, 768] = float:[16, 2]
            ("batch_matmul_2D_042", "batchmatmul_run", ((), 16, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[16384, 768] - [768, 3072] = float:[16384, 3072]
            ("batch_matmul_2D_043", "batchmatmul_run", ((), 16384, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16384, 3072] - [3072, 768] = float:[16384, 768]
            ("batch_matmul_2D_044", "batchmatmul_run", ((), 16384, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16384, 768] - [768, 768] = float:[16384, 768]
            ("batch_matmul_2D_045", "batchmatmul_run", ((), 16384, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 2] - [2, 768] = float:[16, 768]
            ("batch_matmul_2D_046", "batchmatmul_run", ((), 16, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 768] - [768, 768] = float:[16, 768]
            ("batch_matmul_2D_047", "batchmatmul_run", ((), 16, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1, 2] - [2, 768] = float:[1, 768]
            ("batch_matmul_2D_048", "batchmatmul_run", ((), 1, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[1, 768] - [768, 768] = float:[1, 768]
            ("batch_matmul_2D_049", "batchmatmul_run", ((), 1, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[20, 768] - [21128, 768] = float:[20, 21128]
            ("batch_matmul_2D_050", "batchmatmul_run", ((), 20, 21128, 768, (), "float32", False, True, "batch_matmul_output")),


            # float - float:[2048, 768] - [768, 3072] = float:[2048, 3072]
            ("batch_matmul_2D_051", "batchmatmul_run", ((), 2048, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2048, 3072] - [3072, 768] = float:[2048, 768]
            ("batch_matmul_2D_052", "batchmatmul_run", ((), 2048, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2048, 768] - [768, 768] = float:[2048, 768]
            ("batch_matmul_2D_053", "batchmatmul_run", ((), 2048, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[20, 21128] - [21128, 768] = float:[20, 768]
            ("batch_matmul_2D_054", "batchmatmul_run", ((), 20, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[20, 768] - [768, 768] = float:[20, 768]
            ("batch_matmul_2D_055", "batchmatmul_run", ((), 20, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[160, 21128] - [160, 768] = float:[21128, 768]
            ("batch_matmul_2D_056", "batchmatmul_run", ((), 21128, 768, 160, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[20, 21128] - [20, 768] = float:[21128, 768]
            ("batch_matmul_2D_057", "batchmatmul_run", ((), 21128, 768, 20, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[2560, 21128] - [2560, 768] = float:[21128, 768]
            ("batch_matmul_2D_058", "batchmatmul_run", ((), 21128, 768, 2560, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[320, 21128] - [320, 768] = float:[21128, 768]
            ("batch_matmul_2D_059", "batchmatmul_run", ((), 21128, 768, 320, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[40, 21128] - [40, 768] = float:[21128, 768]
            ("batch_matmul_2D_060", "batchmatmul_run", ((), 21128, 768, 40, (), "float32", True, False, "batch_matmul_output")),

            # float - float:[640, 21128] - [640, 768] = float:[21128, 768]
            ("batch_matmul_2D_061", "batchmatmul_run", ((), 21128, 768, 640, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[80, 21128] - [80, 768] = float:[21128, 768]
            ("batch_matmul_2D_062", "batchmatmul_run", ((), 21128, 768, 80, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[2, 12, 128, 64] - [2, 12, 128, 64] = float:[2, 12, 128, 128]
            ("batch_matmul_4D_063", "batchmatmul_run", ((2, 12), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[2, 12, 128, 128] - [2, 12, 128, 64] = float:[2, 12, 128, 64]
            ("batch_matmul_4D_064", "batchmatmul_run", ((2, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2, 128, 1] - [2, 1, 128] = float:[2, 128, 128]
            ("batch_matmul_3D_065", "batchmatmul_run", ((2, ), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2, 768] - [2, 768] = float:[2, 2]
            ("batch_matmul_2D_066", "batchmatmul_run", ((), 2, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[2560, 768] - [21128, 768] = float:[2560, 21128]
            ("batch_matmul_2D_067", "batchmatmul_run", ((), 2560, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[2560, 21128] - [21128, 768] = float:[2560, 768]
            ("batch_matmul_2D_068", "batchmatmul_run", ((), 2560, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2560, 768] - [768, 768] = float:[2560, 768]
            ("batch_matmul_2D_069", "batchmatmul_run", ((), 2560, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[256, 768] - [768, 3072] = float:[256, 3072]
            ("batch_matmul_2D_070", "batchmatmul_run", ((), 2560, 3072, 768, (), "float32", False, False, "batch_matmul_output")),

            # float - float:[256, 3072] - [3072, 768] = float:[256, 768]
            ("batch_matmul_2D_071", "batchmatmul_run", ((), 256, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[256, 768] - [768, 768] = float:[256, 768]
            ("batch_matmul_2D_072", "batchmatmul_run", ((), 256, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2, 2] - [2, 768] = float:[2, 768]
            ("batch_matmul_2D_073", "batchmatmul_run", ((), 2, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[2, 768] - [768, 768] = float:[2, 768]
            ("batch_matmul_2D_074", "batchmatmul_run", ((), 2, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[128, 2] - [128, 768] = float:[2, 768]
            ("batch_matmul_2D_075", "batchmatmul_run", ((), 2, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[16, 2] - [16, 768] = float:[2, 768]
            ("batch_matmul_2D_076", "batchmatmul_run", ((), 2, 768, 16, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[32, 2] - [32, 768] = float:[2, 768]
            ("batch_matmul_2D_077", "batchmatmul_run", ((), 2, 768, 32, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[4, 2] - [4, 768] = float:[2, 768]
            ("batch_matmul_2D_078", "batchmatmul_run", ((), 2, 768, 4, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[8, 2] - [8, 768] = float:[2, 768]
            ("batch_matmul_2D_079", "batchmatmul_run", ((), 2, 768, 8, (), "float32", True, False, "batch_matmul_output")),
            # float - float:[320, 768] - [21128, 768] = float:[320, 21128]
            ("batch_matmul_2D_080", "batchmatmul_run", ((), 320, 21128, 768, (), "float32", False, True, "batch_matmul_output")),

            # float - float:[320, 21128] - [21128, 768] = float:[320, 768]
            ("batch_matmul_2D_081", "batchmatmul_run", ((), 320, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[320, 768] - [768, 768] = float:[320, 768]
            ("batch_matmul_2D_082", "batchmatmul_run", ((), 320, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[32, 12, 128, 64] - [32, 12, 128, 64] = float:[32, 12, 128, 128]
            ("batch_matmul_4D_083", "batchmatmul_run", ((32, 12), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[32, 12, 128, 128] - [32, 12, 128, 64] = float:[32, 12, 128, 64]
            ("batch_matmul_4D_084", "batchmatmul_run", ((32, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[32, 128, 1] - [32, 1, 128] = float:[32, 128, 128]
            ("batch_matmul_3D_085", "batchmatmul_run", ((32,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[32, 768] - [2, 768] = float:[32, 2]
            ("batch_matmul_2D_086", "batchmatmul_run", ((), 32, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[32, 2] - [2, 768] = float:[32, 768]
            ("batch_matmul_2D_087", "batchmatmul_run", ((), 32, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[32, 768] - [768, 768] = float:[32, 768]
            ("batch_matmul_2D_088", "batchmatmul_run", ((), 32, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[40, 768] - [21128, 768] = float:[40, 21128]
            ("batch_matmul_2D_089", "batchmatmul_run", ((), 40, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[40, 21128] - [21128, 768] = float:[40, 768]
            ("batch_matmul_2D_090", "batchmatmul_run", ((), 40, 768, 21128, (), "float32", False, False, "batch_matmul_output")),

            # float - float:[40, 768] - [768, 768] = float:[40, 768]
            ("batch_matmul_2D_091", "batchmatmul_run", ((), 40, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4096, 768] - [768, 3072] = float:[4096, 3072]
            ("batch_matmul_2D_092", "batchmatmul_run", ((), 4096, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4096, 3072] - [3072, 768] = float:[4096, 768]
            ("batch_matmul_2D_093", "batchmatmul_run", ((), 4096, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4096, 768] - [768, 768] = float:[4096, 768]
            ("batch_matmul_2D_094", "batchmatmul_run", ((), 4096, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 12, 128, 64] - [4, 12, 128, 64] = float:[4, 12, 128, 128]
            ("batch_matmul_2D_095", "batchmatmul_run", ((), 4096, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 12, 128, 128] - [4, 12, 128, 64] = float:[4, 12, 128, 64]
            ("batch_matmul_4D_096", "batchmatmul_run", ((4, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 128, 1] - [4, 1, 128] = float:[4, 128, 128]
            ("batch_matmul_3D_097", "batchmatmul_run", ((4,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 768] - [2, 768] = float:[4, 2]
            ("batch_matmul_2D_098", "batchmatmul_run", ((), 4, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[4, 2] - [2, 768] = float:[4, 768]
            ("batch_matmul_2D_099", "batchmatmul_run", ((), 4, 768, 2, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[4, 768] - [768, 768] = float:[4, 768]
            ("batch_matmul_2D_100", "batchmatmul_run", ((), 4, 768, 768, (), "float32", False, False, "batch_matmul_output")),

            # float - float:[512, 768] - [768, 3072] = float:[512, 3072]
            ("batch_matmul_2D_101", "batchmatmul_run", ((), 512, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[512, 3072] - [3072, 768] = float:[512, 768]
            ("batch_matmul_2D_102", "batchmatmul_run", ((), 512, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[512, 768] - [768, 768] = float:[512, 768]
            ("batch_matmul_2D_103", "batchmatmul_run", ((), 512, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[640, 768] - [21128, 768] = float:[640, 21128]
            ("batch_matmul_2D_104", "batchmatmul_run", ((), 640, 21128, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[640, 21128] - [21128, 768] = float:[640, 768]
            ("batch_matmul_2D_105", "batchmatmul_run", ((), 640, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[640, 768] - [768, 768] = float:[640, 768]
            ("batch_matmul_2D_106", "batchmatmul_run", ((), 640, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[64, 12, 128, 64] - [64, 12, 128, 64] = float:[64, 12, 128, 128]
            ("batch_matmul_4D_107", "batchmatmul_run", ((64, 12), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[64, 12, 128, 128] - [64, 12, 128, 64] = float:[64, 12, 128, 64]
            ("batch_matmul_4D_108", "batchmatmul_run", ((64, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[64, 128, 1] - [64, 1, 128] = float:[64, 128, 128]
            ("batch_matmul_3D_109", "batchmatmul_run", ((64,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[80, 768] - [21128, 768] = float:[80, 21128]
            ("batch_matmul_2D_110", "batchmatmul_run", ((), 80, 21128, 768, (), "float32", False, True, "batch_matmul_output")),

            # float - float:[80, 21128] - [21128, 768] = float:[80, 768]
            ("batch_matmul_2D_111", "batchmatmul_run", ((), 80, 768, 21128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[80, 768] - [768, 768] = float:[80, 768]
            ("batch_matmul_2D_112", "batchmatmul_run", ((), 80, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 12, 128, 64] - [8, 12, 128, 64] = float:[8, 12, 128, 128]
            ("batch_matmul_4D_113", "batchmatmul_run", ((8, 12), 128, 128, 64, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[8, 12, 128, 128] - [8, 12, 128, 64] = float:[8, 12, 128, 64]
            ("batch_matmul_4D_114", "batchmatmul_run", ((8, 12), 128, 64, 128, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 128, 1] - [8, 1, 128] = float:[8, 128, 128]
            ("batch_matmul_3D_115", "batchmatmul_run", ((8,), 128, 128, 1, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 768] - [768, 3072] = float:[8192, 3072]
            ("batch_matmul_2D_116", "batchmatmul_run", ((), 8192, 3072, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 3072] - [3072, 768] = float:[8192, 768]
            ("batch_matmul_2D_117", "batchmatmul_run", ((), 8192, 768, 3072, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8192, 768] - [768, 768] = float:[8192, 768]
            ("batch_matmul_2D_118", "batchmatmul_run", ((), 8192, 768, 768, (), "float32", False, False, "batch_matmul_output")),
            # float - float:[8, 768] - [2, 768] = float:[8, 2]
            ("batch_matmul_2D_119", "batchmatmul_run", ((), 8, 2, 768, (), "float32", False, True, "batch_matmul_output")),
            # float - float:[8, 2] - [2, 768] = float:[8, 768]
            ("batch_matmul_2D_120", "batchmatmul_run", ((), 8, 768, 2, (), "float32", False, False, "batch_matmul_output")),

            # float - float:[8, 768] - [768, 768]) = float:[8, 768]
            ("batch_matmul_2D_121", "batchmatmul_run", ((), 8, 768, 768, (), "float32", False, False, "batch_matmul_output")),

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

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def test_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
