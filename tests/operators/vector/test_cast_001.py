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
from base import TestBase
from nose.plugins.attrib import attr
from test_run.cast_run import cast_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_cast_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        # self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("001_cast_test_case_16_dim_1", cast_run, ((16,), "float32", "float16")),
            ("002_cast_test_case_1_23_dim_2", cast_run, ((1, 23), "float16", "float32")),
            ("003_cast_test_case_15_400_dim_2", cast_run, ((15, 400), "float32", "float16")),
            ("006_cast_test_case_16_16_123_dim_3", cast_run, ((16, 16, 123), "float32", "float16")),
            ("007_cast_test_case_4_16_16_5_dim_4", cast_run, ((4, 16, 16, 5), "float32", "float16")),
            ("008_cast_test_case_8_1_128_dim_2", cast_run, ((8, 1, 128), "float32", "float16")),

            # deeplabv3
            # bool --> float32, int 32
            ("cast_test_case_263169_dim_1", cast_run, ((263169,), "bool", "float32")),
            ("cast_test_case_21_dim_1", cast_run, ((21,), "bool", "float32")),
            ("cast_test_case_1052676_dim_1", cast_run, ((1052676,), "bool", "int32")),
            # uint8 --> float32
            ("cast_test_case_328_500_3_dim_3", cast_run, ((328, 500, 3), "uint8", "float32")),
            # uint8 --> int32
            ("cast_test_case_330_500_1_dim_3", cast_run, ((330, 500, 1), "uint8", "int32")),
        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("001_cast_test_case_16_dim_1", cast_run, ((16,), "float16", "float32"), ((16, 0),)),
        ]
        self.testlenet_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("001_cast_test_case_32_dim_4", cast_run, ((1, 3, 32, 32), "float32", "float16")),
            ("002_cast_test_case_16_dim_2", cast_run, ((1, 784), "float16", "float32")),
            ("003_cast_test_case_32_dim_2", cast_run, ((1, 120), "float32", "float16")),
            ("004_cast_test_case_16_dim_2", cast_run, ((120, 784), "float16", "float32")),
            ("005_cast_test_case_32_dim_4", cast_run, ((1, 16, 7, 7), "float32", "float16")),
            ("006_cast_test_case_32_dim_4", cast_run, ((1, 16, 7, 7), "bool", "int32")),
            ("007_cast_test_case_32_dim_4", cast_run, ((1, 3, 32, 32), "bool", "int32")),
            ("002_cast_test_case_16_dim_2", cast_run, ((1, 784), "bool", "int32")),
            ("003_cast_test_case_32_dim_2", cast_run, ((1, 120), "bool", "int32")),
            ("004_cast_test_case_16_dim_2", cast_run, ((120, 784), "bool", "int32")),
        ]
        self.testarg_rpc_cloud = [
            # deeplabv3
            # bool --> float32
            ("cast_test_case_1052676_dim_1", cast_run, ((1052676,), "bool", "float32")),
            # int32 --> float32
            ("cast_test_case_1_dim_1", cast_run, ((1,), "int32", "float32")),
            ("cast_test_case_2_dim_1", cast_run, ((2,), "int32", "float32")),
            # float32 --> int32
            ("cast_test_case_1_dim_1", cast_run, ((1,), "float32", "int32")),
            ("cast_test_case_2_dim_1", cast_run, ((2,), "float32", "int32")),
            # uint8 --> float32
            ("cast_test_case_338_500_3_dim_3", cast_run, ((338, 500, 3), "uint8", "float32")),
            ("cast_test_case_357_500_3_dim_3", cast_run, ((357, 500, 3), "uint8", "float32")),
            ("cast_test_case_332_480_3_dim_3", cast_run, ((332, 480, 3), "uint8", "float32")),
            ("cast_test_case_500_333_3_dim_3", cast_run, ((500, 333, 3), "uint8", "float32")),
            ("cast_test_case_400_400_3_dim_3", cast_run, ((400, 400, 3), "uint8", "float32")),
            ("cast_test_case_442_500_3_dim_3", cast_run, ((442, 500, 3), "uint8", "float32")),
            ("cast_test_case_370_500_3_dim_3", cast_run, ((370, 500, 3), "uint8", "float32")),
            ("cast_test_case_500_174_3_dim_3", cast_run, ((500, 174, 3), "uint8", "float32")),
            ("cast_test_case_383_500_3_dim_3", cast_run, ((383, 500, 3), "uint8", "float32")),
            ("cast_test_case_500_334_3_dim_3", cast_run, ((500, 334, 3), "uint8", "float32")),
            ("cast_test_case_343_500_3_dim_3", cast_run, ((343, 500, 3), "uint8", "float32")),
            ("cast_test_case_351_500_3_dim_3", cast_run, ((351, 500, 3), "uint8", "float32")),
            ("cast_test_case_500_429_3_dim_3", cast_run, ((500, 429, 3), "uint8", "float32")),
            ("cast_test_case_397_500_3_dim_3", cast_run, ((397, 500, 3), "uint8", "float32")),
            ("cast_test_case_356_500_3_dim_3", cast_run, ((356, 500, 3), "uint8", "float32")),
            ("cast_test_case_500_394_3_dim_3", cast_run, ((500, 394, 3), "uint8", "float32")),
            ("cast_test_case_500_191_3_dim_3", cast_run, ((500, 191, 3), "uint8", "float32")),
            ("cast_test_case_361_500_3_dim_3", cast_run, ((361, 500, 3), "uint8", "float32")),
            ("cast_test_case_340_500_3_dim_3", cast_run, ((340, 500, 3), "uint8", "float32")),
            # uint8 --> int32
            ("cast_test_case_455_500_1_dim_3", cast_run, ((455, 500, 1), "uint8", "int32")),
            ("cast_test_case_500_431_1_dim_3", cast_run, ((500, 431, 1), "uint8", "int32")),
            ("cast_test_case_437_480_1_dim_3", cast_run, ((437, 480, 1), "uint8", "int32")),
            ("cast_test_case_500_480_1_dim_3", cast_run, ((500, 480, 1), "uint8", "int32")),
            ("cast_test_case_457_500_1_dim_3", cast_run, ((457, 500, 1), "uint8", "int32")),
            ("cast_test_case_278_500_1_dim_3", cast_run, ((278, 500, 1), "uint8", "int32")),
            ("cast_test_case_458_342_1_dim_3", cast_run, ((458, 342, 1), "uint8", "int32")),
            ("cast_test_case_388_500_1_dim_3", cast_run, ((388, 500, 1), "uint8", "int32")),
            ("cast_test_case_391_500_1_dim_3", cast_run, ((391, 500, 1), "uint8", "int32")),
            ("cast_test_case_468_500_1_dim_3", cast_run, ((468, 500, 1), "uint8", "int32")),
            ("cast_test_case_243_500_1_dim_3", cast_run, ((243, 500, 1), "uint8", "int32")),
            ("cast_test_case_354_500_1_dim_3", cast_run, ((354, 500, 1), "uint8", "int32")),
            ("cast_test_case_245_300_1_dim_3", cast_run, ((245, 300, 1), "uint8", "int32")),
            ("cast_test_case_208_344_1_dim_3", cast_run, ((208, 344, 1), "uint8", "int32")),
            ("cast_test_case_500_313_1_dim_3", cast_run, ((500, 313, 1), "uint8", "int32")),
            ("cast_test_case_352_500_1_dim_3", cast_run, ((352, 500, 1), "uint8", "int32")),
        ]

        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        # self.common_run(self.testarg_rpc_cloud)
        self.common_run(self.testlenet_rpc_cloud)

    def teardown(self):
        # self._log.info("============= {0} Teardown============".format(self.casename))
        return
