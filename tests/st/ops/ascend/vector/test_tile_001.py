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
tile testcases
"""

import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):
    def setup(self):
        case_name = "test_tile_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # shape, dtype, multiples
            ("tile_001", "tile_run", ((1, 26667, 3), "float16", (1, 3, 3)),),
            ("tile_002", "tile_run", ((1, 1, 1280, 1), "float16", (1, 1, 1, 30522)),),

            # SSD
            ("tile_input_1_multiple_8", "tile_run", ((1,), "float16", (8,))),
            ("tile_input_1_multiple_8", "tile_run", ((1,), "float32", (8,))),
            ("tile_input_8_1_multiple_8_8732", "tile_run", ((8, 1), "float16", (8, 8732))),
            ("tile_input_8_1_multiple_8_8732", "tile_run", ((8, 1), "float32", (8, 8732))),
            ("tile_input_2_2_multiple_2_3", "tile_run", ((2, 2), "float16", (2, 3))),
            ("tile_input_2_2_2_multiple_2_3_3", "tile_run", ((2, 2, 2), "float16", (2, 2, 3))),
            ("tile_input_1_multiple_1280", "tile_run", ((1,), "float16", (1280,))),
            ("tile_input_8192_1_multiple_1_1024", "tile_run", ((8192, 1), "float16", (1, 1024))),
            ("tile_input_1280_1_multiple_1_1024", "tile_run", ((1280, 1), "float16", (1, 1024),)),
            ("tile_input_64_128_1_multiple_1_1_1024", "tile_run", ((64, 128, 1), "float16", (1, 1, 1024))),
            ("tile_input_1_multiple_64", "tile_run", ((1,), "float16", (64,))),
            ("tile_input_64_1_multiple_1_2", "tile_run", ((64, 1), "float16", (1, 2))),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, multiples
            ("001_tile_input_2_2_multiple_2_3", "tile_run", ((2, 2), "float32", (2, 3))),
        ]

        self.testarg_rpc_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, multiples
            ("tile_input_1_multiple_8", "tile_run", ((1,), "float16", (8,))),
            ("tile_input_1_multiple_8", "tile_run", ((1,), "float32", (8,))),
            ("tile_input_8_1_multiple_8_8732", "tile_run", ((8, 1), "float16", (8, 8732))),
            ("tile_input_8_1_multiple_8_8732", "tile_run", ((8, 1), "float32", (8, 8732))),
            ("tile_input_64_128_1_multiple_1_1_1024", "tile_run", ((64, 128, 1), "float32", (1, 1, 1024))),
            ("tile_input_1280_1_multiple_1_1024", "tile_run", ((1280, 1), "float32", (1, 1024))),
            ("tile_input_8192_1_multiple_1_1024", "tile_run", ((8192, 1), "float32", (1, 1024))),
            ("tile_input_1280_1_multiple_1_30522", "tile_run", ((1280, 1), "float32", (1, 30522))),
            ("tile_input_1_multiple_1280", "tile_run", ((1,), "float32", (1280,))),
            # float - int32:[64, 128, 1] - [3] = float:[64, 128, 1024]
            ("tile_001", "tile_run", ((64, 128, 1), "float32", (1, 1, 1024))),
            # float - int32:[1280, 1] - [2] = float:[1280, 1024]
            ("tile_002", "tile_run", ((1280, 1), "float32", (1, 1024))),
            # float - int32:[8192, 1] - [2] = float:[8192, 1024]
            ("tile_003", "tile_run", ((8192, 1), "float32", (1, 1024))),
            # float - int32:[1280, 1] - [2] = float:[1280, 30522]
            ("tile_004", "tile_run", ((1280, 1), "float32", (1, 30522))),
            # float - int32:[1] - [1] = float:[1280]
            ("tile_005", "tile_run", ((1,), "float32", (1280,))),
            # float - int32:[1280, 1] - [2] = float:[1280, 21128]
            ("tile_006", "tile_run", ((1280, 1), "float32", (1, 21128))),
            # float - int32:[1280, 1] - [2] = float:[1280, 768]
            ("tile_007", "tile_run", ((1280, 1), "float32", (1, 768))),
            # float - int32:[64, 128, 1] - [3] = float:[64, 128, 768]
            ("tile_008", "tile_run", ((64, 128, 1), "float32", (1, 1, 768))),
            # float - int32:[8192, 1] - [2] = float:[8192, 768]
            ("tile_009", "tile_run", ((8192, 1), "float32", (1, 768))),
            # int32 - int32:[128] - [1] = int32:[16384]
            ("tile_010", "tile_run", ((128,), "int32", (128,))),
            ("tile_011", "tile_run", ((99586,), "float16", (3,))),
            ("tile_012", "tile_run", ((68876, 3), "float16", (2, 1))),
            ("tile_013", "tile_run", ((44017,), "float16", (2,))),
        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, multiples
            ("tile_input_1280_1_multiple_1_30522", "tile_run", ((1280, 1), "float16", (1, 30522))),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
