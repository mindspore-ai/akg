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
tf neg
"""
import os
import pytest
from tests.common.base import TestBase


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_neg_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ('neg_8', "neg_run", ((8,), 'float16')),
            ('neg_64', "neg_run", ((64,), 'float16')),
            ('neg_160', "neg_run", ((160,), 'float16')),
            ('neg_1280', "neg_run", ((1280,), 'float16')),
            ('neg_1280_1024', "neg_run", ((1280, 1024), 'float16')),
            ##('neg_8192_1024', "neg_run", ((8192,1024),'float16'), [(16,16),(1024,1024)]),
            ##('neg_64_128_1024', "neg_run", ((64,128,1024),'float16'), [(1,1),(16,16),(1024,1024)]),
            ('neg_1_128', "neg_run", ((1, 128), 'float16')),
            ('neg_128_128', "neg_run", ((128, 128), 'float16')),
            ('neg_128_256', "neg_run", ((128, 256), 'float16')),
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ('neg_8', "neg_run", ((8,), 'float32'), [(8, 8)]),
            ('neg_64', "neg_run", ((64,), 'float32'), [(64, 64)]),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ('neg_8_fp32', "neg_run", ((8,), 'float32')),
            ('neg_64_fp32', "neg_run", ((64,), 'float32')),
            ('neg_160_fp32', "neg_run", ((160,), 'float32')),
            ('neg_1280_fp32', "neg_run", ((1280,), 'float32')),
            ('neg_1280_1024_fp32', "neg_run", ((1280, 1024), 'float32')),
            ('neg_8192_1024_fp32', "neg_run", ((8192, 1024), 'float32')),
            ('neg_64_128_1024_fp32', "neg_run", ((64, 128, 1024), 'float32')),
            # float:[64] = float:[64]
            ('neg_001_64_fp32', "neg_run", ((64,), 'float32')),
            # float:[8192, 1024] = float:[8192, 1024]
            ('neg_002_8192_1024_fp32', "neg_run", ((8192, 1024), 'float32')),
            # float:[160] = float:[160]
            ('neg_003_160_fp32', "neg_run", ((160,), 'float32')),
            # float:[1280, 1024] = float:[1280, 1024]
            ('neg_004_1280_1024_fp32', "neg_run", ((1280, 1024), 'float32')),
            # float:[1280] = float:[1280]
            ('neg_005_1280_fp32', "neg_run", ((1280,), 'float32')),
            # float:[8] = float:[8]
            ('neg_006_8_fp32', "neg_run", ((8,), 'float32')),
            # float:[64, 128, 1024] = float:[64, 128, 1024]
            ('neg_007_64_128_1024_fp32', "neg_run", ((64, 128, 1024), 'float32')),
            # half:[8192, 768] = half:[8192, 768]
            ('neg_008_8192_768_fp16', "neg_run", ((8192, 768), 'float16')),
            # float:[64] = float:[64]
            ('neg_009_64_fp32', "neg_run", ((64,), 'float32')),
            # float:[64, 128, 768] = float:[64, 128, 768]
            ('neg_010_64_128_768_fp32', "neg_run", ((64, 128, 768), 'float32')),
            # float:[1280, 768] = float:[1280, 768]
            ('neg_011_1280_768_fp32', "neg_run", ((1280, 768), 'float32')),
            # float:[1280] = float:[1280]
            ('neg_012_1280_fp32', "neg_run", ((1280,), 'float32')),
        ]
        self.testarg_level1 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # ('neg_8', "neg_run", ((8,),'float16'), [(8,8)]),
            # ('neg_64', "neg_run", ((64,),'float16'), [(64,64)]),
            # ('neg_160', "neg_run", ((160,),'float16'), [(160,160)]),
            # ('neg_1280', "neg_run", ((1280,),'float16'), [(1280,1280)]),
            # ('neg_1280_1024', "neg_run", ((1280,1024),'float16'), [(16,16),(1024,1024)]),

            ('neg_8192_1024', "neg_run", ((8192, 1024), 'float16')),
            ('neg_64_128_1024', "neg_run", ((64, 128, 1024), 'float16')),

            # ('neg_1_128', "neg_run", ((1,128), 'float16'), [(128,128), (128,128)]),
            # ('neg_128_128', "neg_run", ((128, 128), 'float16'), [(0, 0), (128, 128)]),
            # ('neg_128_256', "neg_run", ((128, 256), 'float16'), [(0, 0), (128, 128)]),
        ]

        return

    @pytest.mark.level0
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
        self.common_run(self.testarg_rpc_cloud)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
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


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
