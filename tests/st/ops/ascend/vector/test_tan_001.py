# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.common.base import TestBase
from tests.common.test_run.tan_run import tan_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_tan_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """set test case """
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("tan_001", tan_run, ((16, 16), "float32")),
            ("tan_002", tan_run, ((13, 23), "float16")),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("tan_001", tan_run, ((4, 4), "int32")),
            ("tan_002", tan_run, ((16, 16), "float32")),
            ("tan_003", tan_run, ((13, 23), "float16")),
        ]
        self.testarg_level2 = [
            # when type is int32 and in mini may cause precision have 1 error.
            ("tan_001", tan_run, ((4, 4), "int32")),
        ]

    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_level2(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level2)

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
