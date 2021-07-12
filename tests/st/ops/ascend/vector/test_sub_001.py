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
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_sub_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("001_sub_2_2_fp16", "sub_run", [(2,), (2,), "float16"]),
            ("002_sub_2_2_int32", "sub_run", [(2,), (2,), "int32"]),
            ("003_sub_64_2_64_2_fp16", "sub_run", [(64, 2), (64, 2), "float16"]),
            ("004_sub_1024_1024_fp16", "sub_run", [(1024,), (1024,), "float16"]),
            ("005_sub_2_1024_2_1024_fp16", "sub_run", [(2, 1024), (2, 1024), "float16"]),
            ("007_sub_30522_30522_fp16", "sub_run", [(30522,), (30522,), "float16"]),
            ("008_sub_1_1024_160_1024_fp16", "sub_run", [(1, 1024), (160, 1024), "float16"]),
            ("009_sub_512_1024_512_1024_fp16", "sub_run", [(512, 1024), (512, 1024), "float16"]),
            ("024_sub_64_16_128_128_64_16_128_128_fp16", "sub_run", [(64, 16, 128, 128), (64, 16, 128, 1), "float16"]),
            ("025_sub_1_64_1_128_128_fp16", "sub_run", [(1,), (64, 1, 128, 128), "float16"]),
            ("026_sub_768_8192_768_fp16", "sub_run", [(768,), (8192, 768), "float16"]),
            # 011 result is not stable
            # ("011_sub_1_1_1_1_8_1_128_128_fp16", "sub_run", [(1, 1, 1, 1), (8, 1, 128, 128), "float16"], [(1,1), (1,1), (1,1), (128,128)]),
            # ("012_sub_1024_1024_1024_1024_fp16", "sub_run", [(1024, 1024), (1024, 1024), "float16"], [(16,16), (1024,1024)]),
            # ("013_sub_1_1_1024_8_128_1024_fp16", "sub_run", [(1, 1, 1024), (8, 128, 1024), "float16"], [(1,1), (1,1), (1024,1024)]),
            # ("014_sub_1_1_1_1_64_1_128_128_fp16", "sub_run", [(1, 1, 1, 1), (64, 1, 128, 128), "float16"], [(1,1), (1,1), (1,1), (128,128)]),
            # ("015_sub_1_1024_1280_1024_fp16", "sub_run", [(1, 1024), (1280, 1024), "float16"], [(1,1), (1024,1024)]),
            # ("016_sub_1280_1024_1280_1_fp16", "sub_run", [(1280, 1024), (1280, 1), "float16"], [(1,1), (1024,1024)]),
            # ("017_sub_4096_1024_4096_1024_fp16", "sub_run", [(4096, 1024), (4096, 1024), "float16"], [(16,16), (1024,1024)]),
            # ("018_sub_1024_4096_1024_4096_fp16", "sub_run", [(1024, 4096), (1024, 4096), "float16"], [(4,4), (4096,4096)]),
            # ("019_sub_1_1024_8192_1024_fp16", "sub_run", [(1, 1024), (8192, 1024), "float16"], [(1,1), (1024,1024)]),
            # ("020_sub_8192_1024_8192_1_fp16", "sub_run", [(8192, 1024), (8192, 1), "float16"], [(1,1), (1024,1024)]),
            # ("021_sub_64_128_1024_64_128_1_fp16", "sub_run", [(64, 128, 1024), (64, 128, 1), "float16"], [(1,1), (1,1), (1024,1024)]),
            # ("022_sub_1_1_1024_64_128_1024_fp16", "sub_run", [(1, 1, 1024), (64, 128, 1024), "float16"], [(1,1), (1,1), (1024,1024)]),
            # ("023_sub_64_16_128_128_64_16_128_1_fp16", "sub_run", [(64, 16, 128, 128), (64, 16, 128, 1), "float16"], [(1,1), (1,1), (1,1), (128,128)]),
            # ("024_sub_30522_1024_30522_1024_fp16", "sub_run", [(30522, 1024), (30522, 1024), "float16"], [(16,16), (1024,1024)]),
            # ("025_sub_8192_4096_8192_4096_fp16", "sub_run", [(8192, 4096), (8192, 4096), "float16"], [(4,4), (4096,4096)]),
            # ("026_sub_1280_30522_1280_30522_fp16", "sub_run", [(1280, 30522), (1280, 30522), "float16"], [(1,1), (16384,16384)]),

        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("001_sub_2_2_fp16", "sub_run", [(2,), (2,), "float32"]),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("007_sub_1_1_fp32", "sub_run", [(1,), (1,), "float32"]),
            ("010_sub_1280_30522_1280_30522_fp32", "sub_run", [(1280, 30522), (1280, 30522), "float32"]),

            ("001_sub_30522_30522_fp32", "sub_run", [(30522,), (30522,), "float32"]),
            ("004_sub_64_128_1024_64_128_1024_fp32", "sub_run", [(64, 128, 1024), (64, 128, 1), "float32"]),
            ("008_sub_1280_1024_1280_1_fp32", "sub_run", [(1280, 1024), (1280, 1), "float32"]),
            ("011_sub_64_2_64_2_fp32", "sub_run", [(64, 2), (64, 2), "float32"]),
            ("012_sub_1024_8_128_1024_fp32", "sub_run", [(1024,), (8, 128, 1024), "float32"]),
            ("013_sub_8192_1024_8192_1_fp32", "sub_run", [(8192, 1024), (8192, 1), "float32"]),
            ("014_sub_1024_64_128_1024_fp32", "sub_run", [(1024,), (64, 128, 1024), "float32"]),
            ("015_sub_2_1024_2_1024_fp32", "sub_run", [(2, 1024), (2, 1024), "float32"]),
            ("016_sub_2_2_fp32", "sub_run", [(2,), (2,), "float32"]),
            ("017_sub_2_2_fp32", "sub_run", [(2,), (2,), "int32"]),
            ("020_sub_1024_160_1024_fp32", "sub_run", [(1024,), (160, 1024), "float32"]),
            ("022_sub_1024_1280_1024_fp32", "sub_run", [(1024,), (1280, 1024), "float32"]),
            ("023_sub_1_8_1_128_128_fp32", "sub_run", [(1,), (8, 1, 128, 128), "float32"]),
            ("024_sub_64_16_128_128_64_16_128_128_fp32", "sub_run", [(64, 16, 128, 128), (64, 16, 128, 1), "float32"]),
            ("025_sub_1_64_1_128_128_fp32", "sub_run", [(1,), (64, 1, 128, 128), "float32"]),
            ("027_sub_768_1280_768_fp32", "sub_run", [(768,), (1280, 768), "float32"]),
            ("028_sub_64_128_768_64_128_1_fp32", "sub_run", [(64, 128, 768), (64, 128, 1), "float32"]),

            # float - float:[30522, 1024] - [30522, 1024] = float:[30522, 1024]
            ("006_sub_30522_1024_30522_1024_fp32", "sub_run", [(30522, 1024), (30522, 1024), "float32"]),
            # float - float:[] - [] = float:[]
            ("007_sub_1_1_fp32", "sub_run", [(1,), (1,), "float32"]),
            # float - float:[] - [8, 1, 128, 128] = float:[8, 1, 128, 128]
            ("023_sub_1_8_1_128_128_fp32", "sub_run", [(1,), (8, 1, 128, 128), "float32"]),
            # float - float:[] - [64, 1, 128, 128] = float:[64, 1, 128, 128]
            ("025_sub_1_64_1_128_128_fp32", "sub_run", [(1,), (64, 1, 128, 128), "float32"]),

            # half - half:[] - [64, 1, 128, 128] = half:[64, 1, 128, 128]
            ("026_sub_1_64_1_128_128_fp32", "sub_run", [(1,), (64, 1, 128, 128), "float16"]),
            # float - float:[2, 768] - [2, 768] = float:[2, 768]
            ("027_sub_2_768_2_768_fp32", "sub_run", [(2, 768), (2, 768), "float32"]),
            # float - float:[768, 768] - [768, 768] = float:[768, 768]
            ("028_sub_768_768_768_768_fp32", "sub_run", [(768, 768), (768, 768), "float32"]),
            # float - float:[1280, 768] - [1280, 1] = float:[1280, 768]
            ("030_sub_1280_768_1280_1_fp32", "sub_run", [(1280, 768), (1280, 1), "float32"]),

            # float - float:[21128] - [21128] = float:[21128]
            ("032_sub_21128_21128_1_fp32", "sub_run", [(21128,), (21128,), "float32"]),
            # float - float:[3072] - [3072] = float:[3072]
            ("033_sub_3072_3072_fp32", "sub_run", [(3072,), (3072,), "float32"]),
            # float - float:[3072, 768] - [3072, 768] = float:[3072, 768]
            ("034_sub_3072_768_3072_768_fp32", "sub_run", [(3072, 768), (3072, 768), "float32"]),
            # float - float:[1280, 21128] - [1280, 21128] = float:[1280, 21128]
            ("035_sub_1280_21128_1280_21128_fp32", "sub_run", [(1280, 21128), (1280, 21128), "float32"]),
            # float - float:[64, 128, 768] - [64, 128, 1] = float:[64, 128, 768]
            ("036_sub_64_128_768_64_128_1_fp32", "sub_run", [(64, 128, 768), (64, 128, 1), "float32"]),

            # float - float:[21128, 768] - [21128, 768] = float:[21128, 768]
            ("037_sub_21128_768_21128_768_fp32", "sub_run", [(21128, 768), (21128, 768), "float32"]),
            # float - float:[768] - [64, 128, 768] = float:[64, 128, 768]
            ("039_sub_768_64_128_768_fp32", "sub_run", [(768,), (64, 128, 768), "float32"]),
            # half - half:[64, 12, 128, 128] - [64, 12, 128, 1] = half:[64, 12, 128, 128]
            ("040_sub_64_12_128_128_64_12_128_1_fp32", "sub_run", [(64, 12, 128, 128), (64, 12, 128, 1), "float16"]),
            # float - float:[] - [] = float:[]
            ("042_sub_1_1_fp32", "sub_run", [(1,), (1,), "float32"]),
            # float - float:[768, 3072] - [768, 3072] = float:[768, 3072]
            ("043_sub_768_3072_768_3072_fp32", "sub_run", [(768, 3072), (768, 3072), "float32"]),
            # float - float:[33, 64] - [33, 64] = float:[33, 64]
            ("045_sub_33_64_33_64_fp32", "sub_run", [(33, 64), (33, 64), "float32"]),
            # half - half:[768] - [8192, 768] = half:[8192, 768]
            ("046_sub_768_8192_768_fp32", "sub_run", [(768,), (8192, 768), "float32"]),
        ]
        self.testarg_level2 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # 011 result is not stable
            ("011_sub_1_1_1_1_8_1_128_128_fp16", "sub_run", [(1, 1, 1, 1), (8, 1, 128, 128), "float16"]),
            ("012_sub_1024_1024_1024_1024_fp16", "sub_run", [(1024, 1024), (1024, 1024), "float16"]),
            ("013_sub_1_1_1024_8_128_1024_fp16", "sub_run", [(1, 1, 1024), (8, 128, 1024), "float16"]),
            ("014_sub_1_1_1_1_64_1_128_128_fp16", "sub_run", [(1, 1, 1, 1), (64, 1, 128, 128), "float16"]),
            ("015_sub_1_1024_1280_1024_fp16", "sub_run", [(1, 1024), (1280, 1024), "float16"]),
            ("016_sub_1280_1024_1280_1_fp16", "sub_run", [(1280, 1024), (1280, 1), "float16"]),
            ("017_sub_4096_1024_4096_1024_fp16", "sub_run", [(4096, 1024), (4096, 1024), "float16"]),
            ("018_sub_1024_4096_1024_4096_fp16", "sub_run", [(1024, 4096), (1024, 4096), "float16"]),
            ("019_sub_1_1024_8192_1024_fp16", "sub_run", [(1, 1024), (8192, 1024), "float16"]),
            ("020_sub_8192_1024_8192_1_fp16", "sub_run", [(8192, 1024), (8192, 1), "float16"]),
            ("021_sub_64_128_1024_64_128_1_fp16", "sub_run", [(64, 128, 1024), (64, 128, 1), "float16"]),
            ("022_sub_1_1_1024_64_128_1024_fp16", "sub_run", [(1, 1, 1024), (64, 128, 1024), "float16"]),
            ("023_sub_64_16_128_128_64_16_128_1_fp16", "sub_run", [(64, 16, 128, 128), (64, 16, 128, 1), "float16"]),
            ("024_sub_30522_1024_30522_1024_fp16", "sub_run", [(30522, 1024), (30522, 1024), "float16"]),
            ("025_sub_8192_4096_8192_4096_fp16", "sub_run", [(8192, 4096), (8192, 4096), "float16"]),
            ("026_sub_1280_30522_1280_30522_fp16", "sub_run", [(1280, 30522), (1280, 30522), "float16"]),
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

    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[0]])

    def test_run_level2(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level2)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
