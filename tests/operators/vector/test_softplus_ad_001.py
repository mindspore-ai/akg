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

"""softplus_ad test case"""

import os

from base import TestBase
from nose.plugins.attrib import attr
from test_run.softplus_ad_run import softplus_ad_run


class Testsoftplus_ad(TestBase):
    def __init__(self):
        """
        testcase preparcondition
        :return:
        """
        casename = "test_akg_softplus_ad_001"
        casepath = os.getcwd()
        super(Testsoftplus_ad, self).__init__(casename, casepath)

    def setup(self):
        self.caseresult = True
        self._log.info("========================{0}  Setup case=================".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("softplus_ad_f16_01", softplus_ad_run, ((32, 16), "float16", "cce_softplus_ad_fp16")),
            ("softplus_ad_f16_02", softplus_ad_run, ((32, 16, 1024), "float16", "cce_softplus_ad_fp16")),
        ]
        return

    @attr(type='rpc_mini')
    @attr(level=0)
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
