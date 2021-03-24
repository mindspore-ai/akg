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
tanh grad test cast
"""
import os
import pytest
from tests.common.base import TestBase


class TestTanhGrad(TestBase):
    def setup(self):
        case_name = "test_akg_tanh_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("tanh_grad_01", "tanh_grad_run", ([64, 1024], "float16"),),
            ("tanh_grad_02", "tanh_grad_run", ([1280, 1024], "float16"),),
            ("tanh_grad_03", "tanh_grad_run", ([128, 4096], "float16"),),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("tanh_grad_01", "tanh_grad_run", ([64, 1024], "float32"),),
        ]

        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            # float:[1280, 1024] = float:[1280, 1024]
            ("tanh_grad_001_fp32", "tanh_grad_run", ([1280, 1024], "float32")),
            # float:[8, 1024] = float:[8, 1024]
            ("tanh_grad_002_fp32", "tanh_grad_run", ([8, 1024], "float32")),
            # float:[64, 1024] = float:[64, 1024]
            ("tanh_grad_003_fp32", "tanh_grad_run", ([64, 1024], "float32")),
            # float:[1024, 4096] = float:[1024, 4096]
            ("tanh_grad_004_fp32", "tanh_grad_run", ([1024, 4096], "float32")),

            # float:[8192, 4096] = float:[8192, 4096]
            ("tanh_grad_005_fp32", "tanh_grad_run", ([8192, 4096], "float32")),
            # float:[160, 1024] = float:[160, 1024]
            ("tanh_grad_006_fp32", "tanh_grad_run", ([160, 1024], "float32")),

            # half - half:[8192, 3072] - [8192, 3072] = half:[8192, 3072]
            ("tanh_grad_007_fp32", "tanh_grad_run", ([8192, 3072], "float16")),
            # float - float:[1280, 768] - [1280, 768] = float:[1280, 768]
            ("tanh_grad_008_fp32", "tanh_grad_run", ([1280, 768], "float32")),
            # float - float:[64, 768] - [64, 768] = float:[64, 768]
            ("tanh_grad_009_fp32", "tanh_grad_run", ([64, 768], "float32")),

        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # ("tanh_grad_01", "tanh_grad_run", ([64, 1024],   "float16"), [(64, 128), (128, 128)]),
            # ("tanh_grad_02", "tanh_grad_run", ([1280, 1024], "float16"), [(64, 0), (128, 128)]),
            # ("tanh_grad_03", "tanh_grad_run", ([128, 4096], "float16"), [(64, 0), (128, 128)]),
            ("tanh_grad_04", "tanh_grad_run", ([8192, 4096], "float16"),),

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
