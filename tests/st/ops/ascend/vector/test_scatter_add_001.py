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
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.scatter_add_run import scatter_add_run


class TestCase(TestBase):
    """define test class"""
    def setup(self):
        case_name = "test_scatter_add_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """set test case """
        self.caseresult = True
        self._log.info("============= %s Setup case============", self.casename)
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("scatter_add_001", scatter_add_run, ((6, ),  (4, 2), "int32", "int32")),
            ("scatter_add_001", scatter_add_run, ((6, 2, 3),  (4, 2), "float16", "int32")),
            # compile error
            # ("scatter_add_001", scatter_add_run, ((3, 2, 2, 7),  (3, 2, 3), "float32", "int32")),
            # ("scatter_add_001", scatter_add_run, ((3, 2, 2, 2, 3),  (4, 2), "float32", "int32")),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("scatter_add_001", scatter_add_run, ((6, ),  (4, 2), "int32", "int32")),
            ("scatter_add_001", scatter_add_run, ((6, 2, 3),  (4, 2), "float16", "int32")),
            ("scatter_add_001", scatter_add_run, ((3, 2, 2, 7),  (3, 2, 3), "float32", "int32")),
        ]
        self.testarg_level2 = [
            # when shape larger than 5D, may have problem.
            ("scatter_add_001", scatter_add_run, ((3, 2, 2, 2, 3, 7),  (4, 2), "float32", "int32")),
            ("scatter_add_001", scatter_add_run, ((3, 2, 2, 2, 3, 7, 2),  (4, ), "float32", "int32")),
        ]

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= %s Setup case============", self.casename)
