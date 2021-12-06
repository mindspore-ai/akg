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
from tests.common.test_run.ascend.split_run import split_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_split_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # fail cases when split Tensor to shape (1,)
            # ("split_001",split_run,((4,1), (1,3), 0, "float16")),
            # ("split_001",split_run,((6,1), (1,1,1,1,1,1), 0, "int32")),
            # ("split_001",split_run,((4,1), (1,3), 0, "int32")),
            ("split_002", split_run, ((4, 1), 2, 0, "int32")),
            ("split_003", split_run, ((38714,), (30522, 8192), 0, "int32")),
            ("split_004", split_run, ((2, 7), 7, 1, "float16")),
            ("split_005", split_run, ((2, 7), 7, 1, "float16")),
            ("split_006", split_run, ((1760,), (1600, 160), 0, "float16")),
            ("split_007", split_run, ((80, 16), 2, 0, "float16")),
            ("split_008", split_run, ((40, 32), 2, 1, "float16")),
            ("split_009", split_run, ((64, 16), 2, 0, "float16")),
            ("split_010", split_run, ((32, 32), 2, 1, "float16")),
            ("split_011", split_run, ((112, 16), (16, 32, 64), 0, "float16")),
            ("split_012", split_run, ((16, 112), (16, 32, 64), 1, "float16")),
            ("split_013", split_run, ((2, 112, 16), (16, 32, 64), 1, "float16")),
            ("split_014", split_run, ((2, 112, 16, 16), (16, 32, 64), 1, "float16")),
            ("split_015", split_run, ((1, 111, 16, 16), (16, 31, 64), 1, "float16")),
            ("split1", split_run, ((8, 8732, 6), (5776, 2166, 600, 150, 36, 4), 1, "float32")),
            ("split2", split_run, ((8, 8732, 6), (4, 36, 150, 600, 2166, 5776), 1, "float32")),
            ("split3", split_run, ((8, 8732, 4), (5776, 2166, 600, 150, 36, 4), 1, "float32")),
            ("split4", split_run, ((8, 8732, 4), (4, 36, 150, 600, 2166, 5776), 1, "float32")),
            ("split5", split_run, ((8, 8732, 6), (5776, 2166, 600, 150, 36, 4), 1, "float16")),
            ("split6", split_run, ((8, 8732, 6), (4, 36, 150, 600, 2166, 5776), 1, "float16")),
            ("split7", split_run, ((8, 8732, 4), (5776, 2166, 600, 150, 36, 4), 1, "float16")),
            ("split8", split_run, ((8, 8732, 4), (4, 36, 150, 600, 2166, 5776), 1, "float16")),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # fail cases when split Tensor to shape (1,)
            # ("split_001",split_run,((4,1), (1,3), 0, "float16")),
            # ("split_001",split_run,((4,1), (1,3), 0, "int32")),
            ("split_001", split_run, ((2, 4), 2, 0, "int32")),
            ("split_01", split_run, ((2, 2, 6), 3, 2, "float16")),
            ("split_02", split_run, ((4, 3), 2, 0, "float16")),
            ("split_004", split_run, ((38714, 16), (30520, 8194), 0, "float16")),
            # fail cases on rpc_cloud because of 1024 alignment, can be avoided by set_dim if necessary
            # ("split_014",split_run,((2, 112, 16, 16), (16, 32, 64), 1, "float16")),
            # ("split_016",split_run,((38714, 1024), (30522, 8192), 0, "float32")),
            ("split_017", split_run, ((38714, 1024), (30522, 8192), 0, "int32")),
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

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
