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
tf_transpose
"""
import os
import pytest
from tests.common.base import TestBase


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_tf_five2four"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,opfuncname,testRunArgs, dimArgs
            ##
            ("five2four_001", "five2four_run", ([1, 16, 16, 16], "float16", 'NCHW', "float16")),
            ("five2four_002", "five2four_run", ([2, 32, 4, 4], "float16", 'NCHW', "float16")),
            ("five2four_003", "five2four_run", ([2, 42, 4, 4], "float16", 'NCHW', "float16")),
            ("five2four_004", "five2four_run", ([1, 60, 15, 15], "float16", 'NCHW', "float16")),
            ("five2four_005", "five2four_run", ([2, 4, 4, 32], "float16", 'NHWC', "float16")),
            ("five2four_006", "five2four_run", ([2, 4, 4, 42], "float16", 'NHWC', "float16")),
            ("five2four_007", "five2four_run", ([2, 32, 4, 4], "float32", 'NCHW', "float16")),
            ("five2four_008", "five2four_run", ([2, 4, 4, 32], "float32", 'NHWC', "float16")),

            # resetnet 50:
            ("five2four_009", "five2four_run", ([32, 2048, 1, 1], "float16", 'NCHW', "float16")),
            ("five2four_010", "five2four_run", ([1001, 2048, 1, 1], "float16", 'NCHW', "float16")),
            ("five2four_011", "five2four_run", ([1, 1001, 1, 1], "float16", 'NCHW', "float16")),
            ("five2four_012", "five2four_run", ([32, 1001, 1, 1], "float16",'NCHW', "float16")),

            # lenet
            ("five2four_013", "five2four_run", ([1, 6, 15, 15], "float16", 'NCHW', "float16")),
            ("five2four_014", "five2four_run", ([1, 16, 7, 7], "float16", 'NCHW', "float16")),
        ]

        self.testarg_rpc_cloud = [
            ("five2four_001", "five2four_run", ([1, 16, 16, 16], "float16", 'NCHW', "float32")),
            ("five2four_002", "five2four_run", ([2, 32, 4, 4], "float16", 'NCHW', "float32")),
            ("five2four_004", "five2four_run", ([2, 4, 4, 32], "float16", 'NHWC', "float32")),
            # resetnet 50:
            ("five2four_006", "five2four_run", ([32, 2048, 1, 1], "float32", 'NCHW', "float32")),
            ("five2four_007", "five2four_run", ([1001, 2048, 1, 1], "float32", 'NCHW', "float32")),
            ("five2four_008", "five2four_run", ([1, 1001, 1, 1], "float32", 'NCHW', "float32")),
            ("five2four_009", "five2four_run", ([32, 1001, 1, 1], "float32", 'NCHW', "float32")),

            #  test find some problem and need to solve(20191016 21:00):
            # ("five2four_003", five2four_run, ([2, 42, 4, 4], "float16",'NCHW', "float32")),
            # ("five2four_005", five2four_run, ([2, 4, 4, 42], "float16",'NHWC', "float32")),
            # ("five2four_010", five2four_run, ([32, 2048, 1, 1], "float32", 'NCHW', "float16")),
            # ("five2four_011", five2four_run, ([1001, 2048, 1, 1], "float32", 'NCHW', "float16")),
        ]
        self.testarg_level1 = [
            ("five2four_001", "five2four_run", ([1, 64, 16, 16], "float16", 'NCHW', "float16")),
            ("five2four_002", "five2four_run", ([1, 64, 15, 15], "float16", 'NCHW', "float16")),
            ("five2four_003", "five2four_run", ([1, 64, 121, 16], "float16", 'NCHW', "float16")),
            ("five2four_004", "five2four_run", ([3, 50, 62, 60], "float16", 'NCHW', "float16")),
            ("five2four_005", "five2four_run", ([20, 5001, 1, 1], "float16", 'NCHW', "float16")),
            ("five2four_006", "five2four_run", ([33, 5008, 1, 1], "float16", 'NCHW', "float16")),
            ("five2four_007", "five2four_run", ([1, 12280, 1, 1], "float16", 'NCHW', "float16")),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

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
