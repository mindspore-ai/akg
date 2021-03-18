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

"""testcase for concat op"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.concat_run import concat_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_concat_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #testflag,opfuncname,testRunArgs, dimArgs
            ('concat_001', concat_run, ([[1], [1]], 'int32', 0)),
            ('concat_002', concat_run, ([[2, 1], [2, 1]], 'int32', 0)),
            ('concat_003', concat_run, ([[30522, ], [8192, ]], 'int32', 0)),
            ('concat_005', concat_run, ([[1600, ]], 'float16', 0)),
            ('concat_006', concat_run, ([[1600, ], [160, ]], 'float16', 0)),
            ('concat_007', concat_run, ([[40, 16], [40, 16]], 'float16', 0)),
            ('concat_008', concat_run, ([[40, 16], [40, 16]], 'float16', 1)),
            ('concat_009', concat_run, ([[32, 16], [32, 16]], 'float16', 0)),
            ('concat_010', concat_run, ([[32, 16], [32, 16]], 'float16', 1)),
            ('concat_011', concat_run, ([[16, 16], [32, 16], [64, 16]], 'float16', 0)),
            ('concat_012', concat_run, ([[16, 16], [16, 32], [16, 64]], 'float16', 1)),
            ('concat_013', concat_run, ([[16, 16, 16], [16, 32, 16], [16, 64, 16]], 'float16', 1)),
            ('concat_014', concat_run, ([[2, 16, 16, 16], [2, 32, 16, 16], [2, 64, 16, 16]], 'float16', 1)),
            ('concat_015', concat_run, ([[1, 16, 16, 16], [1, 31, 16, 16], [1, 64, 16, 16]], 'float16', 1)),
            # SSD testcases
            ('concat1', concat_run, ([[8, 5776, 6], [8, 2166, 6], [8, 600, 6], [8, 150, 6], [8, 36, 6], [8, 4, 6]], 'float32', 1)),
            ('concat2', concat_run, ([[8, 4, 6], [8, 36, 6], [8, 150, 6], [8, 600, 6], [8, 2166, 6], [8, 5776, 6]], 'float32', 1)),
            ('concat3', concat_run, ([[8, 5776, 4], [8, 2166, 4], [8, 600, 4], [8, 150, 4], [8, 36, 4], [8, 4, 4]], 'float32', 1)),
            ('concat4', concat_run, ([[8, 4, 4], [8, 36, 4], [8, 150, 4], [8, 600, 4], [8, 2166, 4], [8, 5776, 4]], 'float32', 1)),
            ('concat5', concat_run, ([[8, 5776, 6], [8, 2166, 6], [8, 600, 6], [8, 150, 6], [8, 36, 6], [8, 4, 6]], 'float16', 1)),
            ('concat6', concat_run, ([[8, 4, 6], [8, 36, 6], [8, 150, 6], [8, 600, 6], [8, 2166, 6], [8, 5776, 6]], 'float16', 1)),
            ('concat7', concat_run, ([[8, 5776, 4], [8, 2166, 4], [8, 600, 4], [8, 150, 4], [8, 36, 4], [8, 4, 4]], 'float16', 1)),
            # With random problem
            #('concat8', concat_run, ([[8, 4, 4], [8, 36, 4], [8, 150, 4], [8, 600, 4], [8, 2166, 4], [8, 5776, 4]], 'float16', 1)),
        ]
        self.testarg_rpc_cloud = [
            #testflag,opfuncname,testRunArgs, dimArgs
            ('concat_004', concat_run, ([[30522, 1024], [8192, 1024]], 'float16', 0)),
            ('concat_016', concat_run, ([[30522, 1024], [8192, 1024]], 'float32', 0)),
            ('concat_017', concat_run, ([[30522, 1024], [8192, 1024]], 'int32', 0)),
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
