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

import datetime
import os

from base import TestBase
from nose.plugins.attrib import attr
from test_run.reduce_logsumexp_ad_run import reduce_logsumexp_ad_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def __init__(self):
        """
        testcase preparcondition
        :return:
        """
        casename = "test_akg_reduce_logsumexp_ad"
        casepath = os.getcwd()
        super(TestCase, self).__init__(casename, casepath)

    def setup(self):
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # shape, axis, keepdims, dtype, attrs
            ("reduce_logsumexp_ad_1", reduce_logsumexp_ad_run, ((32, 64, 32), "float16", (1,), True, "cce_reduce_logsumexp_ad_fp16")),
            ("reduce_logsumexp_ad_2", reduce_logsumexp_ad_run, ((4, 128, 1024), "float16", (0, 1), True, "cce_reduce_logsumexp_ad_fp16")),
            ("reduce_logsumexp_ad_4", reduce_logsumexp_ad_run, ((8, 256), "float16", (0,), False, "cce_reduce_logsumexp_ad_fp16")),
            ("reduce_logsumexp_ad_5", reduce_logsumexp_ad_run, ((1024,), "float16", (0,), True, "cce_reduce_logsumexp_ad_fp16")),
        ]

    @attr(type='rpc_mini')
    @attr(level=0)
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
