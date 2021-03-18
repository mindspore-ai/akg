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


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_triangle"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            ("lower_triangle_001", "triangle_run", ([350], 0.0, True, "float16"), ((1, 1), (350, 1))),
            #("lower_triangle_002", "triangle_run", ([350], 0.0, True, "float32")),

            #("lower_triangle_003", "triangle_run", ([1024], 0.0, True, "float16")),
            #("lower_triangle_004", "triangle_run", ([1024], 0.0, True, "float32")),

            #("lower_triangle_005", "triangle_run", ([350], 0.0, False, "float16")),
            #("lower_triangle_006", "triangle_run", ([350], 0.0, False, "float32")),

            ("lower_triangle_005", "triangle_run", ([1024], 0.0, False, "float16"), ((1, 1), (1024, 1))),
            #("lower_triangle_006", "triangle_run", ([1024], 0.0, False, "float32")),
        ]

        self.testarg_rpc_cloud = [
        ]

        self.testarg_level1 = [
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

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

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
