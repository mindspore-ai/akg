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
        case_name = "test_akg_square_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag, opfuncname, testRunArgs, dimArgs
            # shape, dtype, kernel_name, attrs
            ("square_001", "square_run", ((4096,), "float16", "cce_mod_fp16")),
            ("square_002", "square_run", ((1024, 4096), "float16", "cce_mod_fp16")),
            ("square_003", "square_run", ((8192, 1024), "float16", "cce_mod_fp16")),
            ("square_004", "square_run", ((8, 128, 1024), "float16", "cce_mod_fp16")),
            ("square_005", "square_run", ((1280, 1024), "float16", "cce_mod_fp16")),
            ("square_006", "square_run", ((30522,), "float16", "cce_mod_fp16")),
            ("square_007", "square_run", ((160, 1024), "float16", "cce_mod_fp16")),
            ("square_008", "square_run", ((64, 128, 1024), "float16", "cce_mod_fp16")),
            ("square_009", "square_run", ((1024, 1024), "float16", "cce_mod_fp16")),
            ("square_010", "square_run", ((1024,), "float16", "cce_mod_fp16")),
            ("square_011", "square_run", ((2,), "float16", "cce_mod_fp16")),
        ]
        self.testarg_cloud = [
            ## testflag, opfuncname, testRunArgs, dimArgs
            # shape, dtype, kernel_name, attrs
            #("square_012", "square_run", ((2,), "float32", "cce_mod_fp16")),
        ]

        self.testarg_rpc_cloud = [
            ## testflag, opfuncname, testRunArgs, dimArgs
            # shape, dtype, kernel_name, attrs
            # float:[4096]
            ("square_001_fp32", "square_run", ((4096,), "float32", "cce_mod_fp32")),
            # float:[1280, 1024]
            ("square_002_fp32", "square_run", ((1280, 1024), "float32", "cce_mod_fp32")),
            # float:[1024, 1024]
            ("square_003_fp32", "square_run", ((1024, 1024), "float32", "cce_mod_fp32")),
            # float:[2, 1024]
            ("square_004_fp32", "square_run", ((2, 1024), "float32", "cce_mod_fp32")),
            # float:[4096, 1024]
            ("square_005_fp32", "square_run", ((4096, 1024), "float32", "cce_mod_fp32")),
            # float:[8192, 4096]
            ("square_006_fp32", "square_run", ((8192, 4096), "float32", "cce_mod_fp32")),
            # float:[1024]
            ("square_007_fp32", "square_run", ((1024,), "float32", "cce_mod_fp32")),
            # float:[1024, 4096]
            ("square_008_fp32", "square_run", ((1024, 4096), "float32", "cce_mod_fp32")),
            # float:[30522]
            ("square_009_fp32", "square_run", ((30522,), "float32", "cce_mod_fp32")),
            # float:[30522, 1024]
            ("square_010_fp32", "square_run", ((30522, 1024), "float32", "cce_mod_fp32")),
            # float:[2]
            ("square_011_fp32", "square_run", ((2,), "float32", "cce_mod_fp32")),
            # float:[512, 1024]
            ("square_012_fp32", "square_run", ((512, 1024), "float32", "cce_mod_fp32")),
            # float:[768, 3072] = float:[768, 3072]
            ("square_013_fp32", "square_run", ((512, 1024), "float32", "cce_mod_fp32")),
            # half:[8192, 3072] = half:[8192, 3072]
            ("square_014_fp32", "square_run", ((8192, 3072), "float16", "cce_mod_fp32")),
            # float:[1280, 768] = float:[1280, 768]
            ("square_015_fp32", "square_run", ((1280, 768), "float32", "cce_mod_fp32")),
            # float:[768, 768] = float:[768, 768]
            ("square_016_fp32", "square_run", ((768, 768), "float32", "cce_mod_fp32")),
            # float:[3072] = float:[3072]
            ("square_017_fp32", "square_run", ((3072,), "float32", "cce_mod_fp32")),
            # float:[3072, 768] = float:[3072, 768]
            ("square_018_fp32", "square_run", ((512, 1024), "float32", "cce_mod_fp32")),
            # float:[21128, 768] = float:[21128, 768]
            ("square_019_fp32", "square_run", ((21128, 768), "float32", "cce_mod_fp32")),
            # float:[21128] = float:[21128]
            ("square_020_fp32", "square_run", ((21128,), "float32", "cce_mod_fp32")),
            # float:[2] = float:[2]
            ("square_021_fp32", "square_run", ((2,), "float32", "cce_mod_fp32")),
            # float:[33, 64] = float:[33, 64]
            ("square_022_fp32", "square_run", ((33, 64), "float32", "cce_mod_fp32")),
            # float:[768] = float:[768]
            ("square_023_fp32", "square_run", ((768,), "float32", "cce_mod_fp32")),
            # float:[2, 768] = float:[2, 768]
            ("square_024_fp32", "square_run", ((2, 768), "float32", "cce_mod_fp32")),
        ]
        self.testarg_level1 = [
            ## testflag, opfuncname, testRunArgs, dimArgs
            # shape, dtype, kernel_name, attrs
            ("square_001", "square_run", ((30522, 1024), "float16", "cce_mod_fp16")),
        ]
        return

    @pytest.mark.level2
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
