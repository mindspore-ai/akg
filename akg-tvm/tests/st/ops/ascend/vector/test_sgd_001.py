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
import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.sgd_run import sgd_run


class TestSgd(TestBase):
    def setup(self):
        case_name = "test_akg_sgd_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= %s Setup case============", self.casename)
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ("sgd_01", sgd_run, ((1, 2), "float16", False, 0.0, 1.0)),
            ("sgd_02", sgd_run, ((3, 2), "float32", True, 0.0, 0.0)),
            ("sgd_03", sgd_run, ((5, 2), "float32", False, 1.0, 2.0)),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("sgd_01", sgd_run, ((1, 2), "float16", False, 0.0, 1.0)),
            ("sgd_02", sgd_run, ((2, 2), "float16", False, 0.0, 1.3)),
            ("sgd_03", sgd_run, ((3, 2), "float32", True, 0.0, 0.0)),
            ("sgd_04", sgd_run, ((3, 2), "float32", True, 0.0, 0.5)),
            ("sgd_05", sgd_run, ((4, 2), "float16", False, 1.0, 0.5)),
            ("sgd_06", sgd_run, ((2, 2), "float32", False, 1.0, 2.0)),
        ]
        return

    @pytest.mark.level1
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
        self._log.info("============= %s Setup case============", self.casename)
