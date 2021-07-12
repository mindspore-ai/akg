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
from tests.common.test_run.batch_reindex_layer_run import batch_reindex_layer_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_batch_reindex_layer_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("batch_reindex_layer_001", batch_reindex_layer_run, ((3, 4, 5), (1, 2, 0, 1), "int32")),
            ("batch_reindex_layer_002", batch_reindex_layer_run, ((2, 3), (1, 0, 1), "float16")),
            ("batch_reindex_layer_003", batch_reindex_layer_run, ((2, 3, 16, 1024), (0, 0, 1), "int8")),
            ("batch_reindex_layer_004", batch_reindex_layer_run, ((8, 3, 16), (2, 0, 1), "float32")),
            ("batch_reindex_layer_005", batch_reindex_layer_run, ((8, 24, 42), (2, 2, 2, 2, 2), "uint8")),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("batch_reindex_layer_006", batch_reindex_layer_run, ((3, 4, 5), (1, 2, 0, 1), "int32")),
            ("batch_reindex_layer_007", batch_reindex_layer_run, ((2, 3), (1, 0, 1), "float16")),
            ("batch_reindex_layer_008", batch_reindex_layer_run, ((2, 3, 16, 1024), (0, 0, 1), "int8")),
            ("batch_reindex_layer_009", batch_reindex_layer_run, ((8, 3, 16), (2, 0, 1), "float32")),
            ("batch_reindex_layer_010", batch_reindex_layer_run, ((8, 24, 42), (2, 2, 2, 2, 2), "uint8")),
        ]
        return

    @pytest.mark.level2
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
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
