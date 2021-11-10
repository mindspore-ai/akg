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
from tests.common.base import TestBase
from tests.common.test_run.ascend.logsigmoid_ad_run import logsigmoid_ad_run



############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def __init__(self):
        """
        testcase preparcondition
        :return:
        """
        casename = "test_akg_logsigmoid_ad"
        casepath = os.getcwd()
        super(TestCase, self).__init__(casename, casepath)

    def setup(self):
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("logsigmoid_ad_01", logsigmoid_ad_run, [(1, 128), "float16", "cce_logsigmoid_ad_fp16"], [(1, 1), (128, 128)]),
            ("logsigmoid_ad_02", logsigmoid_ad_run, [(128, 128), "float16", "cce_logsigmoid_ad_fp16"], [(128, 128), (128, 128)]),
            ("logsigmoid_ad_03", logsigmoid_ad_run, [(32, 128), "float16", "cce_logsigmoid_ad_fp16"], [(32, 32), (128, 128)]),
            ("logsigmoid_ad_04", logsigmoid_ad_run, [(128, 32), "float16", "cce_logsigmoid_ad_fp16"], [(128, 128), (32, 32)]),
            ("logsigmoid_ad_05", logsigmoid_ad_run, [(32, 32), "float16", "cce_logsigmoid_ad_fp16"], [(32, 32), (32, 32)]),
            ("logsigmoid_ad_06", logsigmoid_ad_run, [(384, 32), "float16", "cce_logsigmoid_ad_fp16"], [(384, 384), (32, 32)]),

        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("logsigmoid_ad_01", logsigmoid_ad_run, [(1, 128), "float16", "cce_logsigmoid_ad_fp16"], [(1, 1), (128, 128)]),
        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
