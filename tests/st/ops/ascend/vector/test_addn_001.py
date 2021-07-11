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
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:

"""
import os
import pytest
from tests.common.base import TestBase

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_addn_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("003_bc_addn_input_1_1_256_48_2_dim_4", "addn_run", ((1, 1, 256, 48), "float16", 3)),
            ("004_bc_addn_input_1_1_256_48_2_dim_4", "addn_run", ((1, 1, 256, 48), "float16", 4)),
            ("005_bc_addn_input_1_1_256_48_2_dim_4", "addn_run", ((1, 1, 256, 48), "float16", 5)),
            #("002_addn_input_8192_1024_2_dim_2","addn_run",(((8192, 1024), (8192, 1024)), "float16"), ((32, 32), (1024, 1024))),
            #("003_addn_input_8_128_1024_2_dim_3","addn_run",(((8, 128, 1024), (8, 128, 1024)), "float16"), ((1,1), (32, 32), (1024, 1024))),
            # ("004_addn_input_64_128_1024_2_dim_3","addn_run",(((64, 128, 1024), (64, 128, 1024)), "float16"), ((1,1), (32, 32), (1024, 1024))),
            ("0014_addn_input_64_128_1024_2_dim_3", "addn_run", ((16, 16), "float16", 4), ((16, 16), (16, 16))),
            #("005_addn_input_1280_1024_3_dim_2","addn_run",(((1280, 1024), (1280, 1024), (1280, 1024)), "float16"), ((32, 32), (1024, 1024))),
            #("006_addn_input_8192_1024_3_dim_2","addn_run",(((8192, 1024), (8192, 1024), (8192, 1024)), "float16"), ((16, 16), (1024, 1024))),
            #("007_addn_input_8192_4096_3_dim_2","addn_run",(((8192, 4096), (8192, 4096), (8192, 4096)), "float16"), ((4, 4), (4096, 4096))),
            #("008_addn_input_64_128_1024_3_dim_3","addn_run",(((64, 128, 1024), (64, 128, 1024),(64, 128, 1024)), "float16"), ((1,1), (16, 16), (1024, 1024))),
            #("009_addn_input_8192_1024_4_dim_2","addn_run",(((8192, 1024), (8192, 1024), (8192, 1024), (8192, 1024)), "float16"), ((8, 8), (1024, 1024))),
            ("011_addn_input_1_1_256_48_2_dim_4", "addn_run", ((1, 1, 256, 48), "float16", 2)),
            ("011_addn_input_1_1_1280_256_2_dim_4", "addn_run", ((1, 1, 1280, 256), "float16", 2)),
            ("012_addn_input_1_1_128_128_2_dim_4", "addn_run", ((1, 1, 128, 128), "float16", 2)),
            ("013_addn_input_1_1_304_256_2_dim_4", "addn_run", ((1, 1, 304, 256), "float16", 2)),
            ("015_addn_input_1_1_256_256_2_dim_4", "addn_run", ((1, 1, 256, 256), "float16", 2)),
            ("016_addn_input_3_3_32_64_2_dim_4", "addn_run", ((3, 3, 32, 64), "float16", 2)),
            ("017_addn_input_1_1_256_21_2_dim_4", "addn_run", ((1, 1, 256, 21), "float16", 2)),
            ("018_addn_input_1_1_64_128_2_dim_4", "addn_run", ((1, 1, 64, 128), "float16", 2)),
            ("019_addn_input_1_1_256_728_2_dim_4", "addn_run", ((1, 1, 256, 728), "float16", 2)),
            ("020_addn_input_3_3_3_32_2_dim_4", "addn_run", ((3, 3, 3, 32), "float16", 2)),
            ("021_addn_input_1_1_128_256_2_dim_4", "addn_run", ((1, 1, 128, 256), "float16", 2)),
        ]

        self.testarg_rpc_cloud = [
            # float - float:[1280, 1024] - [1280, 1024] = float:[1280, 1024]
            ("addn_001_input_fp32", "addn_run", ((1280, 1024), "float32", 2)),
            # float - float:[64, 128, 1024] - [64, 128, 1024] = float:[64, 128, 1024]
            ("addn_002_input_fp32", "addn_run", ((64, 128, 1024), "float32", 2)),
            # float - float:[8, 128, 1024] - [8, 128, 1024] = float:[8, 128, 1024]
            ("addn_003_input_fp32", "addn_run", ((8, 128, 1024), "float32", 2)),
            # float - float - float:[8192, 1024] - [8192, 1024] - [8192, 1024] = float:[8192, 1024]
            ("addn_004_input_fp32", "addn_run", ((8192, 1024), "float32", 3)),
            # float - float:[8192, 1024] - [8192, 1024] = float:[8192, 1024]
            ("addn_005_input_fp32", "addn_run", ((8192, 1024), "float32", 2)),
            # float - float - float - float:[8192, 1024] - [8192, 1024] - [8192, 1024] - [8192, 1024] = float:[8192, 1024]
            ("addn_006_input_fp32", "addn_run", ((8192, 1024), "float32", 4)),
            # float - float - float:[64, 128, 1024] - [64, 128, 1024] - [64, 128, 1024] = float:[64, 128, 1024]
            ("addn_007_input_fp32", "addn_run", ((64, 128, 1024), "float32", 3)),
            # float - float - float:[8192, 4096] - [8192, 4096] - [8192, 4096] = float:[8192, 4096]
            ("addn_008_input_fp32", "addn_run", ((8192, 4096), "float32", 3)),
            # float - float - float:[1280, 1024] - [1280, 1024] - [1280, 1024] = float:[1280, 1024]
            ("addn_009_input_fp32", "addn_run", ((1280, 1024), "float32", 3)),
            # half - half - half - half:[8192, 768] - [8192, 768] - [8192, 768] - [8192, 768] = half:[8192, 768]
            ("addn_010_input_fp32", "addn_run", ((8192, 768), "float16", 4)),
            # half - half:[8192, 768] - [8192, 768] = half:[8192, 768]
            ("addn_011_input_fp32", "addn_run", ((8192, 768), "float16", 2)),
            # float - float:[64, 128, 768] - [64, 128, 768] = float:[64, 128, 768]
            ("addn_012_input_fp32", "addn_run", ((8192, 768), "float32", 2)),
            # float - float:[21128, 768] - [21128, 768] = float:[21128, 768]
            ("addn_013_input_fp32", "addn_run", ((21128, 768), "float32", 2)),
            # half - half - half:[8192, 3072] - [8192, 3072] - [8192, 3072] = half:[8192, 3072]
            ("addn_014_input_fp32", "addn_run", ((8192, 3072), "float16", 3)),
            # half - half:[64, 12, 128, 64] - [64, 12, 128, 64] = half:[64, 12, 128, 64]
            ("addn_015_input_fp32", "addn_run", ((64, 12, 128, 64), "float16", 2)),
            # float - float:[1280, 768] - [1280, 768] = float:[1280, 768]
            ("addn_016_input_fp32", "addn_run", ((1280, 768), "float32", 2)),
            # float - float:[8192, 768] - [8192, 768] = float:[8192, 768]
            ("addn_017_input_fp32", "addn_run", ((8192, 768), "float32", 2)),
            # float - float - float:[64, 128, 768] - [64, 128, 768] - [64, 128, 768] = float:[64, 128, 768]
            ("addn_018_input_fp32", "addn_run", ((64, 128, 768), "float32", 3)),
            # half - half:[64, 12, 128, 128] - [64, 12, 128, 128] = half:[64, 12, 128, 128]
            ("addn_019_input_fp32", "addn_run", ((8192, 768), "float16", 2)),
            # float - float - float:[1280, 768] - [1280, 768] - [1280, 768] = float:[1280, 768]
            ("addn_020_input_fp32", "addn_run", ((1280, 768), "float32", 3)),
        ]
        self.testarg_level1 = [
            #testflag,opfuncname,testRunArgs, dimArgs
            #("001_addn_input_1280_1024_2_dim_2","addn_run",((128, 1024), "float16", 2), ((32, 32), (1024, 1024))),
            #("002_addn_input_8192_1024_2_dim_2","addn_run",((8192, 1024), "float16", 2), ((32, 32), (1024, 1024))),
            #("003_addn_input_8_128_1024_2_dim_3","addn_run",((8, 128, 1024), "float16", 2), ((1,1), (32, 32), (1024, 1024))),
            #("004_addn_input_64_128_1024_2_dim_3","addn_run",(((64, 128, 1024), (64, 128, 1024)), "float16"), ((1,1), (32, 32), (1024, 1024))),
            #("005_addn_input_1280_1024_3_dim_2","addn_run",((1280, 1024), "float16", 3), ((32, 32), (1024, 1024))),
            #("006_addn_input_8192_1024_3_dim_2","addn_run",((8192, 1024), "float16", 3), ((16, 16), (1024, 1024))),
            #("007_addn_input_8192_4096_3_dim_2","addn_run",((8192, 4096), "float16", 3), ((4, 4), (4096, 4096))),
            #("008_addn_input_64_128_1024_3_dim_3","addn_run",((64, 128, 1024), "float16", 4), ((1,1), (16, 16), (1024, 1024))),
            #("009_addn_input_8192_1024_4_dim_2","addn_run",((8192, 1024), "float16", 4), ((8, 8), (1024, 1024))),
            ("010_addn_input_4_129_129_128_2_dim_4", "addn_run", ((4, 129, 129, 128), "float16", 2)),
            ("012_addn_input_4_33_33_728_2_dim_4", "addn_run", ((4, 33, 33, 728), "float16", 2)),
            ("013_addn_input_1_1_1024_1024_2_dim_4", "addn_run", ((1, 1, 1024, 1024), "float16", 2)),
            ("016_addn_input_1_1_2048_256_2_dim_4", "addn_run", ((1, 1, 2048, 256), "float16", 2)),
            ("017_addn_input_1_1_1536_2048_2_dim_4", "addn_run", ((1, 1, 1536, 2048), "float16", 2)),
            ("019_addn_input_4_129_129_256_2_dim_4", "addn_run", ((4, 129, 129, 256), "float16", 2)),
            ("020_addn_input_4_65_65_256_2_dim_4", "addn_run", ((4, 65, 65, 256), "float16", 2)),
            ("022_addn_input_4_33_33_2048_5_dim_4", "addn_run", ((4, 33, 33, 2048), "float16", 5)),
            ("026_addn_input_1_1_1536_1536_2_dim_4", "addn_run", ((1, 1, 1536, 1536), "float16", 2)),
            ("027_addn_input_1_1_728_1024_dim_4", "addn_run", ((1, 1, 728, 1024), "float16", 2)),
            ("029_addn_input_1_1_1024_1536_2_dim_4", "addn_run", ((1, 1, 1024, 1536), "float16", 2)),
            ("031_addn_input_4_257_257_64_2_dim_4", "addn_run", ((4, 257, 257, 64), "float16", 2)),
            ("032_addn_input_1_1_728_728_2_dim_4", "addn_run", ((1, 1, 728, 728), "float16", 2)),
        ]
        # Set all shape in cloud
        self.testarg_5d_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("001_addn_input_32_64_14_14_16_dim_5", "addn_run", ((32, 64, 14, 14, 16), "float16", 2)),
            ("002_addn_input_32_128_7_7_16_dim_5", "addn_run", ((32, 128, 7, 7, 16), "float16", 2)),
            ("003_addn_input_32_16_56_56_16_dim_5", "addn_run", ((32, 16, 56, 56, 16), "float16", 2)),
            ("004_addn_input_32_32_28_28_16_dim_5", "addn_run", ((32, 32, 28, 28, 16), "float16", 2)),
            ("005_addn_input_32_4_56_56_16_dim_5", "addn_run", ((32, 4, 56, 56, 16), "float16", 2)),
            ("006_addn_input_32_64_14_14_16_dim_5", "addn_run", ((32, 64, 14, 14, 16), "float32", 2)),
            ("007_addn_input_32_128_7_7_16_dim_5", "addn_run", ((32, 128, 7, 7, 16), "float32", 2)),
            ("008_addn_input_32_16_56_56_16_dim_5", "addn_run", ((32, 16, 56, 56, 16), "float32", 2)),
            ("009_addn_input_32_32_28_28_16_dim_5", "addn_run", ((32, 32, 28, 28, 16), "float32", 2)),
            ("010_addn_input_32_4_56_56_16_dim_5", "addn_run", ((32, 4, 56, 56, 16), "float32", 2)),

        ]

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_5d_rpc_cloud)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
