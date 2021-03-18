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
from tests.common.test_run.minimum_ad_run import minimum_ad_run


class TestCase(TestBase):
    def setup(self):
        """set test case """
        case_name = "test_minimum_ad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= %s Setup case============", self.casename)
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("minimum_ad_001", minimum_ad_run, ((2, 2, 2), "int32", True, True)),
            ("minimum_ad_002", minimum_ad_run, ((2, 2), "float16", True, False)),
            ("minimum_ad_003", minimum_ad_run, ((2, 3, 3, 4), "int32", False, True)),

        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("minimum_ad_001", minimum_ad_run, ((2, 3, 3, 4), "float32", False, True)),
            ("minimum_ad_002", minimum_ad_run, ((2, 2, 1), "float16", True, True)),
            ("minimum_ad_003", minimum_ad_run, ((2, 3, 3, 4), "int32", False, True)),
            ("minimum_ad_004", minimum_ad_run, ((16, 16), "float16", True, False)),
            ("minimum_ad_005", minimum_ad_run, ((8, 16), "int32", True, True)),
        ]

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
        self._log.info("============= %s Setup case============", self.casename)