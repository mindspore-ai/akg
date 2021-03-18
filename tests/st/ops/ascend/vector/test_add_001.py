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
################################################
"""
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.add_run import add_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_add_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #testflag,opfuncname,testRunArgs, dimArgs
            ("add_1_1_int32", add_run, ([1], [1], "int32", "cce_add_int32")),
            ("add_2_1_int32", add_run, ([2], [1], "int32", "cce_add_int32")),
            ("001_add_2_2_fp16", add_run, ([2], [2], "float16", "cce_add_fp16", 3.0)),
            ("001b_add_2_2_fp16", add_run, ([2], [2], "float16", "cce_add_fp16", 2.0)),
            ("002_add_1024_1024_fp16", add_run, ([1024], [1024], "float16", "cce_add_fp16")),
            ("004_add_30522_30522_fp16", add_run, ([30522], [30522], "float16", "cce_add_fp16")),
            ("005_add_2_1024_2_1024_fp16", add_run, ([2, 1024], [2, 1024], "float16", "cce_add_fp16")),
            ("006_add_160_1024_160_1024_fp16", add_run, ([160, 1024], [160, 1024], "float16", "cce_add_fp16")),
            ("007_add_512_1024_512_1024_fp16", add_run, ([512, 1024], [512, 1024], "float16", "cce_add_fp16")),
            ("010_add_4096_1024_4096_1024_fp16", add_run, ([4096, 1024], [4096, 1024], "float16", "cce_add_fp16")),
            # manual scheduling tests, man sch work for all the shapes but including only two tests in the CI
            ("012_add_2_2_fp16", add_run, ([2], [2], "float16", "cce_add_fp16", 3.0), ([2, 2],), False),
            #("013_add_160_1024_160_1024_fp16", add_run, ([160, 1024], [160, 1024], "float16", "cce_add_fp16", 1.0), ([16, 16], [1024, 1024]), False),
            ("014_add_160_1024_160_1024_fp16", add_run, ([160, 1024], [160, 1024], "float16", "cce_add_fp16"), ([20, 20], [1024, 1024]), False),

            # deeplabv3
            ("add", add_run, ([1, 1, 192, 64], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([576, 750, 1], [1], "int32", "cce_add_int32")),
        ]
        self.testarg_level1 = [
            #testflag,opfuncname,testRunArgs, dimArgs
            ("101_add_8192_1024_8192_1024_fp16", add_run, ([8192, 1024], [8192, 1024], "float16", "cce_add_fp16")),
            ("102_add_30522_1024_30522_1024_fp16", add_run, ([30522, 1024], [30522, 1024], "float16", "cce_add_fp16")),
            ("103_add_1024_4096_1024_4096_fp16", add_run, ([1024, 4096], [1024, 4096], "float16", "cce_add_fp16")),
            ("104_add_8192_4096_8192_4096_fp16", add_run, ([8192, 4096], [8192, 4096], "float16", "cce_add_fp16")),
            ("105_add_8_128_1024_8_128_1024_fp16", add_run, ([8, 128, 1024], [8, 128, 1024], "float16", "cce_add_fp16")),
            ("106_add_64_128_1024_64_128_1024_fp16", add_run, ([64, 128, 1024], [64, 128, 1024], "float16", "cce_add_fp16")),
            ("107_add_1_128_1024_8_128_1024_fp16", add_run, ([1, 128, 1024], [8, 128, 1024], "float16", "cce_add_fp16")),
            ("108_add_1_128_1024_64_128_1024_fp16", add_run, ([1, 128, 1024], [64, 128, 1024], "float16", "cce_add_fp16")),
            ("109_add_8_16_128_128_8_1_128_128_fp16", add_run, ([8, 16, 128, 128], [8, 1, 128, 128], "float16", "cce_add_fp16")),
            ("110_add_64_16_128_128_64_1_128_128_fp16", add_run, ([64, 16, 128, 128], [64, 1, 128, 128], "float16", "cce_add_fp16")),
            ("111_add_1_1_fp16", add_run, ([1], [1], "float16", "cce_add_fp16")),
            ("112_add_2_1_fp16", add_run, ([2], [1], "float16", "cce_add_fp16")),
            ("113_add_1024_1_fp16", add_run, ([1024], [1], "float16", "cce_add_fp16")),
            ("115_add_30522_1_fp16", add_run, ([30522], [1], "float16", "cce_add_fp16")),
            ("116_add_160_1_1_fp16", add_run, ([160, 1], [1], "float16", "cce_add_fp16")),
            ("117_add_1024_1_1_fp16", add_run, ([1024, 1], [1], "float16", "cce_add_fp16")),
            ("119_add_8192_1_1_fp16", add_run, ([8192, 1], [1], "float16", "cce_add_fp16")),
            ("120_add_2_1024_1_fp16", add_run, ([2, 1024], [1], "float16", "cce_add_fp16")),
            ("121_add_512_1024_1_fp16", add_run, ([512, 1024], [1], "float16", "cce_add_fp16")),
            ("122_add_1024_1024_1_fp16", add_run, ([1024, 1024], [1], "float16", "cce_add_fp16")),
            ("123_add_4096_1024_1_fp16", add_run, ([4096, 1024], [1], "float16", "cce_add_fp16")),
            ("125_add_30522_1024_1_fp16", add_run, ([30522, 1024], [1], "float16", "cce_add_fp16")),
            ("126_add_8_128_1_1_fp16", add_run, ([8, 128, 1], [1], "float16", "cce_add_fp16")),
            ("127_add_64_128_1_1_fp16", add_run, ([64, 128, 1], [1], "float16", "cce_add_fp16")),
            ("128_add_1_160_1024_fp16", add_run, ([1], [160, 1024], "float16", "cce_add_fp16")),
            ("129_add_1_1280_1024_fp16", add_run, ([1], [1280, 1024], "float16", "cce_add_fp16")),
            ("130_add_1_8192_1024_fp16", add_run, ([1], [8192, 1024], "float16", "cce_add_fp16")),
            ("131_add_1_1024_4096_fp16", add_run, ([1], [1024, 4096], "float16", "cce_add_fp16")),
            ("133_add_1_64_128_1024_fp16", add_run, ([1], [64, 128, 1024], "float16", "cce_add_fp16")),
            ("134_add_1_64_16_128_128_fp16", add_run, ([1], [64, 16, 128, 128], "float16", "cce_add_fp16")),

            # manual scheduling tests, it works for all the shapes in this testfile, but only including few in the CI
            ("135_add_64_16_128_128_64_1_128_128_fp16", add_run, ([64, 16, 128, 128], [64, 1, 128, 128], "float16", "cce_add_fp16", 1.0), ([1, 1], [1, 1], [128, 128], [128, 128]), False),
            ("136_add_2_1_fp16", add_run, ([2], [1], "float16", "cce_add_fp16", 1.0), ([2, 2],), False),
            #("137_add_512_1024_1_fp16", add_run, ([512, 1024], [1], "float16", "cce_add_fp16", 1.0), ([1, 1], [1024, 1024]), False),
            ("138_add_1_64_128_1024_fp16", add_run, ([1], [64, 128, 1024], "float16", "cce_add_fp16", 1.0), ([1, 1], [128, 128], [128, 128]), False),

            # deeplabv3
            ("add", add_run, ([1, 1, 256, 3], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 576, 160], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([666, 1000, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 960, 320], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 960, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 192, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([875, 656, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 32, 192], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 96, 576], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 160, 96], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1000, 750, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 129, 129, 128], [1, 129, 129, 128], "float32", "cce_add_fp32")),
            ("add", add_run, ([562, 750, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 16, 96], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 384, 96], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([576, 750, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 32, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1000, 626, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 3, 3, 160], [1, 3, 3, 160], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 65, 65, 256], [1, 65, 65, 256], "float32", "cce_add_fp32")),
            ("add", add_run, ([750, 1000, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([513, 625, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 3, 3, 96], [1, 3, 3, 96], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 5, 5, 32], [1, 5, 5, 32], "float32", "cce_add_fp32")),
            ("add", add_run, ([513, 750, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([513, 513, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([513, 513, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 64, 384], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([591, 875, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([4, 129, 129, 128], [4, 129, 129, 128], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 32, 16], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 24, 144], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([513, 875, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([627, 750, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1], [4, 33, 33, 256], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 3, 32], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 96, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 256, 256], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([550, 625, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 320, 256], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 9, 9, 24], [1, 9, 9, 24], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 960, 160], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 33, 33, 3], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 144, 32], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 192, 32], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 144, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([625, 513, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([21], [21], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 256, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 576, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([4, 33, 33, 728], [4, 33, 33, 728], "float32", "cce_add_fp32")),
            ("add", add_run, ([582, 875, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 3, 3, 64], [1, 3, 3, 64], "float32", "cce_add_fp32")),
            ("add", add_run, ([656, 875, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 384, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 33, 33, 1024], [1, 33, 33, 1024], "float32", "cce_add_fp32")),
            ("add", add_run, ([4, 65, 65, 256], [4, 65, 65, 256], "float32", "cce_add_fp32")),
            ("add", add_run, ([830, 1000, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([750, 513, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 96, 24], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 512, 256], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 144, 48], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 304, 256], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 144, 2], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([624, 875, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([582, 875, 3], [1, 1, 3], "float32", "cce_add_fp32")),
            ("add", add_run, ([3, 3, 304, 1], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([4, 33, 33, 1024], [4, 33, 33, 1024], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 384, 64], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 33, 33, 728], [1, 33, 33, 728], "float32", "cce_add_fp32")),
            ("add", add_run, ([1, 1, 576, 96], [1], "float32", "cce_add_fp32")),
            ("add", add_run, ([750, 1000, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([513, 750, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([875, 656, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([591, 875, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([513, 875, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([1000, 626, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([624, 875, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([513, 625, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([550, 625, 1], [1], "int32", "cce_add_int32")),
            ("add", add_run, ([625, 513, 1], [1], "int32", "cce_add_int32")),
        ]

        self.testarg_rpc_cloud = [
            # resnet50 5D:
            ("200_add_fp16", add_run, ([32, 16, 56, 56, 16], [32, 16, 56, 56, 16], "float16", "cce_add_fp16")),
            ("201_add_fp16", add_run, ([32, 32, 28, 28, 16], [32, 32, 28, 28, 16], "float16", "cce_add_fp16")),
            ("202_add_fp16", add_run, ([32, 64, 14, 14, 16], [32, 64, 14, 14, 16], "float16", "cce_add_fp16")),
            ("203_add_fp16", add_run, ([32, 128, 7, 7, 16], [32, 128, 7, 7, 16], "float16", "cce_add_fp16")),
            ("200_add_fp32", add_run, ([32, 16, 56, 56, 16], [32, 16, 56, 56, 16], "float32", "cce_add_fp32")),
            ("201_add_fp32", add_run, ([32, 32, 28, 28, 16], [32, 32, 28, 28, 16], "float32", "cce_add_fp32")),
            ("202_add_fp32", add_run, ([32, 64, 14, 14, 16], [32, 64, 14, 14, 16], "float32", "cce_add_fp32")),
            ("203_add_fp32", add_run, ([32, 128, 7, 7, 16], [32, 128, 7, 7, 16], "float32", "cce_add_fp32")),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def test_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
