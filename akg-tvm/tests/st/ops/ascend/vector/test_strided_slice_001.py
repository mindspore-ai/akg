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


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_strided_slice_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            ("test_strided_slice1", "strided_slice_run", ((4,), [1], [3], [1], 0, 0, 0, 0, 0, "float16")),
            ("test_strided_slice2", "strided_slice_run",
             ((8, 128, 1024), [0, 0, 0], [4, 128, 512], [1, 1, 1], 0, 0, 0, 0, 0, "float16")),
            ("test_strided_slice3", "strided_slice_run", (
                (5, 2, 3, 4, 5, 6, 7), [0, 1, 1, 0, 0, 0, 0], [0, 2, 0, 0, 2, 0, 0], [1, 1, 1, 1, 1, 1, 1], 17, 5, 32, 72,
                2, "float16")),
            ("test_strided_slice4", "strided_slice_run", ((20, 100, 40), [10, 11, 12], [19, 100, 39], [2, 3, 4], 0, 0, 0, 0, 0, "float16")),
        ]
        self.testarg_cloud = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            ("test_strided_slice1", "strided_slice_run", ((4,), [1], [3], [1], 0, 0, 0, 0, 0, "float32")),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs

            # fail for now and need to find why:
            # ("test_strided_slice1_int32", "strided_slice_run", ((1,), [0], [0], [1], 0, 0, 0, 0, 0, "int32"), ((1, 1),)),
            # ("test_strided_slice4_int32", "strided_slice_run", ((2,), [1], [1], [-1], 0, 0, 0, 0, 0, "int32"), ((2, 2),)),

            ("test_strided_slice2_int32", "strided_slice_run", ((2,), [0], [1], [1], 0, 0, 0, 0, 0, "int32")),
            ("test_strided_slice3_int32", "strided_slice_run", ((1,), [0], [1], [1], 0, 0, 0, 0, 0, "int32")),
            ("test_strided_slice5_fp32", "strided_slice_run",
             ((64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            ("test_strided_slice6_fp32", "strided_slice_run",
             ((8, 128, 1024), [0, 0, 0], [8, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[1] - [1] - [1] - [1] = int32:[0]
            ("strided_slice_005_fp32", "strided_slice_run",
             ((64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),

            # int32 - int32 - int32 - int32:[2] - [1] - [1] - [1] = int32:[1]
            ("strided_slice_001_fp32", "strided_slice_run",
             ((2,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),

            ("strided_slice_002_fp32", "strided_slice_run",
             ((2,), [1, ], [2, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[1] - [1] - [1] - [1] = int32:[1]
            ("strided_slice_003_fp32", "strided_slice_run",
             ((1,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32:[64, 128, 1024] - [3] - [3] - [3] = float:[64, 1, 1024]
            ("strided_slice_004_fp32", "strided_slice_run",
             ((64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # float - int32 - int32 - int32:[8, 128, 1024] - [3] - [3] - [3] = float:[8, 1, 1024]
            ("strided_slice_005_fp32", "strided_slice_run",
             ((8, 128, 1024), [0, 0, 0], [8, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[2] - [1] - [1] - [1] = int32:[]
            ("strided_slice_006_fp32", "strided_slice_run",
             ((8, 128, 1024), [0, 0, 0], [8, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),

            # float - int32 - int32 - int32:[64, 128, 768] - [3] - [3] - [3] = float:[64, 1, 768]
            ("strided_slice_007_fp32", "strided_slice_run",
             ((64, 128, 768), [0, 0, 0], [64, 1, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            ("strided_slice_007_fp32", "strided_slice_run",
             ((64, 128, 768), [0, 1, 0], [64, 2, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            ("strided_slice_007_fp32", "strided_slice_run",
             ((64, 128, 768), [0, 2, 0], [64, 3, 768], [1, 1, 1], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[2] - [1] - [1] - [1] = int32:[1]
            ("strided_slice_008_fp32", "strided_slice_run",
             ((2,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[1] - [1] - [1] - [1] = int32:[0]
            ("strided_slice_009_fp32", "strided_slice_run",
             ((1,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[1] - [1] - [1] - [1] = int32:[1]
            ("strided_slice_010_fp32", "strided_slice_run",
             ((1,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
            # int32 - int32 - int32 - int32:[2] - [1] - [1] - [1] = int32:[]
            ("strided_slice_011_fp32", "strided_slice_run",
             ((2,), [0, ], [1, ], [1, ], 0, 0, 0, 0, 0, "float32")),
        ]
        self.testarg_level1 = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            ("test_strided_slice1", "strided_slice_run", ((64, 128, 1024), [32, 64, 0], [64, 128, 4], [4, 1, 2], 0, 0, 0, 0, 0, "float16")),
            ("test_strided_slice2", "strided_slice_run", ((64, 128, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float16")),
            ("test_strided_slice3", "strided_slice_run", ((64, 512, 1024), [0, 0, 0], [64, 1, 1024], [1, 1, 1], 0, 0, 0, 0, 0, "float16")),
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
        self.common_run([self.testarg_rpc_cloud[0]])

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
