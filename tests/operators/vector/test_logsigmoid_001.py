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
from base import TestBase
from nose.plugins.attrib import attr
from test_run.logsigmoid_run import logsigmoid_run



############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def __init__(self):
        """
        testcase preparcondition
        :return:
        """
        casename = "test_akg_logsigmoid"
        casepath = os.getcwd()
        super(TestCase, self).__init__(casename, casepath)

    def setup(self):
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("logsigmoid_01", logsigmoid_run, [(1, 128), "float16", "cce_logsigmoid_fp16"], [(1, 1), (128, 128)]),
            ("logsigmoid_02", logsigmoid_run, [(128, 128), "float16", "cce_logsigmoid_fp16"], [(128, 128), (128, 128)]),
            ("logsigmoid_03", logsigmoid_run, [(32, 128), "float16", "cce_logsigmoid_fp16"], [(32, 32), (128, 128)]),
            ("logsigmoid_04", logsigmoid_run, [(128, 32), "float16", "cce_logsigmoid_fp16"], [(128, 128), (32, 32)]),
            ("logsigmoid_05", logsigmoid_run, [(32, 32), "float16", "cce_logsigmoid_fp16"], [(32, 32), (32, 32)]),
            ("logsigmoid_06", logsigmoid_run, [(384, 32), "float16", "cce_logsigmoid_fp16"], [(384, 384), (32, 32)]),

        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("logsigmoid_01", logsigmoid_run, [(1, 128), "float16", "cce_logsigmoid_fp16"], [(1, 1), (128, 128)]),
        ]
        return

    @attr(type='rpc_mini')
    @attr(level=0)
    def test_run(self):
        self.common_run(self.testarg)

    @attr(type='aicmodel')
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    @attr(type='rpc_cloud')
    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    @attr(level=1)
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
