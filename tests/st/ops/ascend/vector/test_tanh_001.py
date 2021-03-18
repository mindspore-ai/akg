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

import os
import pytest
from tests.common.base import TestBase


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_tanh_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ('tanh_8_1024', "tanh_run", ((8, 1024), 'float16'),),
            ('tanh_64_1024', "tanh_run", ((64, 1024), 'float16'),),
            ('tanh_160_1024', "tanh_run", ((160, 1024), 'float16'),),
            ('tanh_1280_1024', "tanh_run", ((1280, 1024), 'float16'),),
            ##('tanh_1024_4096', "tanh_run", ((1024,4096),'float16'), [(4,4),(4096,4096)]),
            ##('tanh_8192_4096', "tanh_run", ((8192,4096),'float16'), [(4,4),(4096,4096)]),
            ##('tanh_4_4096', "tanh_run", ((4, 4096), 'float16'), [(1, 1), (128, 128)]),
            ('tanh_4_1024', "tanh_run", ((4, 1024), 'float16'),),
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # ('tanh_8_1024', "tanh_run", ((8, 1024), 'float32'),),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ('tanh_8_1024_fp32', "tanh_run", ((8, 1024), 'float32')),
            ('tanh_64_1024_fp32', "tanh_run", ((64, 1024), 'float32')),
            ('tanh_160_1024_fp32', "tanh_run", ((160, 1024), 'float32')),
            ('tanh_1280_1024_fp32', "tanh_run", ((1280, 1024), 'float32')),
            ('tanh_1024_4096_fp32', "tanh_run", ((1024, 4096), 'float32')),
            ('tanh_8192_4096_fp32', "tanh_run", ((8192, 4096), 'float32')),

            # float:[1280, 1024] = float:[1280, 1024]
            ('tanh_001_1280_1024_fp32', "tanh_run", ((1280, 1024), 'float32')),
            # float:[8, 1024] = float:[8, 1024]
            ('tanh_002_8_1024_fp32', "tanh_run", ((8, 1024), 'float32')),
            # float:[64, 1024] = float:[64, 1024]
            ('tanh_003_64_1024_fp32', "tanh_run", ((64, 1024), 'float32')),
            # float:[1024, 4096] = float:[1024, 4096]
            ('tanh_004_1024_4096_fp32', "tanh_run", ((1024, 4096), 'float32')),
            # float:[8192, 4096] = float:[8192, 4096]
            ('tanh_005_8192_4096_fp32', "tanh_run", ((8192, 4096), 'float32')),
            # float:[160, 1024] = float:[160, 1024]
            ('tanh_006_160_1024_fp32', "tanh_run", ((160, 1024), 'float32')),
            # float:[64, 768] = float:[64, 768]
            ('tanh_007_64_768_fp32', "tanh_run", ((64, 768), 'float32')),
            # float:[1280, 768] = float:[1280, 768]
            ('tanh_008_1280_768_fp32', "tanh_run", ((1280, 768), 'float32')),
            # half:[8192, 3072] = half:[8192, 3072]
            ('tanh_009_8192_3072_fp32', "tanh_run", ((8192, 3072), 'float32')),
        ]
        self.testarg_level1 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # ('tanh_8_1024', "tanh_run", ((8,1024),'float16'), [(8,8),(1024,1024)]),
            # ('tanh_64_1024', "tanh_run", ((64,1024),'float16'), [(16,16),(1024,1024)]),
            # ('tanh_160_1024', "tanh_run", ((160,1024),'float16'), [(16,16),(1024,1024)]),
            # ('tanh_1280_1024', "tanh_run", ((1280,1024),'float16'), [(16,16),(1024,1024)]),
            ('tanh_1024_4096', "tanh_run", ((1024, 4096), 'float16'),),
            ('tanh_8192_4096', "tanh_run", ((8192, 4096), 'float16'),),
            ('tanh_4_4096', "tanh_run", ((4, 4096), 'float16'),),
            # ('tanh_4_1024', "tanh_run", ((4, 1024), 'float16'), [(1, 1), (128, 128)]),
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
        self.common_run([self.testarg_rpc_cloud[0]])

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
