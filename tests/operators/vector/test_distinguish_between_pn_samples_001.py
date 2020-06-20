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
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.distinguish_between_pn_samples_run import distinguish_between_pn_samples_run
#from print_debug import print_debug

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_distinguish_between_pn_samples_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("01_distinguish_between_pn_samples_01", distinguish_between_pn_samples_run, ((8, 4718, 16), 0.618, "float16")),
            ("01_distinguish_between_pn_samples_02", distinguish_between_pn_samples_run, ((8, 4718, 16), 0.618, "float32")),
            ("01_distinguish_between_pn_samples_03", distinguish_between_pn_samples_run, ((8, 4718, 16), 0.222, "float16")),
            ("01_distinguish_between_pn_samples_04", distinguish_between_pn_samples_run, ((8, 4718, 16), 0.222, "float32")),
            ("01_distinguish_between_pn_samples_05", distinguish_between_pn_samples_run, ((8, 4718, 16), 0.999, "float16")),
            ("01_distinguish_between_pn_samples_06", distinguish_between_pn_samples_run, ((8, 4718, 16), 0.999, "float32")),
        ]
        self.testarg_2 = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("01_distinguish_between_pn_samples_01", distinguish_between_pn_samples_run, ((8, 8732, 16), 0.618, "float16")),
            ("01_distinguish_between_pn_samples_02", distinguish_between_pn_samples_run, ((8, 8732, 16), 0.618, "float32")),
            ("01_distinguish_between_pn_samples_03", distinguish_between_pn_samples_run, ((8, 8732, 16), 0.222, "float16")),
            ("01_distinguish_between_pn_samples_04", distinguish_between_pn_samples_run, ((8, 8732, 16), 0.222, "float32")),
            ("01_distinguish_between_pn_samples_05", distinguish_between_pn_samples_run, ((8, 8732, 16), 0.999, "float16")),
            ("01_distinguish_between_pn_samples_06", distinguish_between_pn_samples_run, ((8, 8732, 16), 0.999, "float32")),

        ]

        self.testarg_aic_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("01_distinguish_between_pn_samples_01", distinguish_between_pn_samples_run, ((8, 18, 16), 0.618, "float16")),
            ("01_distinguish_between_pn_samples_02", distinguish_between_pn_samples_run, ((8, 18, 16), 0.618, "float32")),
        ]

        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.level2
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_2(self):

        self.common_run(self.testarg)

    @pytest.mark.aic_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_ci(self):
        self.common_run(self.testarg_aic_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
