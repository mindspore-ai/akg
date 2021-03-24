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
from tests.common.test_run.realdiv_run import realdiv_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_realdiv_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_realdiv", realdiv_run, ((8, 512), (1, 512), "float16", "cce_realdiv_fp16")),
            ("002_realdiv", realdiv_run, ((2, 512), (3, 2, 512), "float16", "cce_realdiv_fp16")),
            ("003_realdiv", realdiv_run,((8,),(1,),"float16","cce_realdiv_fp16")),
            ("004_realdiv", realdiv_run, ((8,), (1,), "float32", "cce_realdiv_fp16")),
            ("005_realdiv", realdiv_run, ((2,), (2,), "float16", "cce_realdiv_fp16"), ),
            ("006_realdiv", realdiv_run, ((1024,), (1024,), "float16", "cce_realdiv_fp16"), ),
            # ("007_realdiv", realdiv_run, ((4096,), (4096,), "float16", "cce_realdiv_fp16"), ),
            # ("008_realdiv", realdiv_run, ((30522, ), (30522, ), "float16", "cce_realdiv_fp16"), ),
            # ("009_realdiv", realdiv_run, ((2, 1024), (2, 1024), "float16", "cce_realdiv_fp16"), ),
            # ("010_realdiv", realdiv_run, ((512, 1024), (512, 1024), "float16", "cce_realdiv_fp16"), ),
            # ("011_realdiv", realdiv_run, ((1024, 1024), (1024, 1024), "float16", "cce_realdiv_fp16"), ),
            # ("012_realdiv", realdiv_run, ((4096, 1024), "float16", "cce_realdiv_fp16"), ),
            # ("013_realdiv", realdiv_run, ((1024, 4096), "float16", "cce_realdiv_fp16"), ),
        ]
        self.testarg_cloud = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_realdiv_2", realdiv_run, ((2,), (2,), "float32", "cce_realdiv_fp16"), ),
        ]
        self.testarg_level1 = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            # ("001_realdiv_2", realdiv_run,((2,),"float16","cce_realdiv_fp16"), ),
            # ("002_realdiv_1024", realdiv_run,((1024,),"float16","cce_realdiv_fp16"), ),
            # ("003_realdiv_4096", realdiv_run,((4096,), "float16","cce_realdiv_fp16"), ),
            # ("004_realdiv_30522", realdiv_run, ((30522, ), "float16", "cce_realdiv_fp16"), ),
            # ("005_realdiv_2_1024", realdiv_run, ((2, 1024), "float16", "cce_realdiv_fp16"), ),
            # ("006_realdiv_512_1024", realdiv_run, ((512, 1024), "float16", "cce_realdiv_fp16"), ),
            # ("007_realdiv_1024_1024", realdiv_run, ((1024, 1024), "float16", "cce_realdiv_fp16"),),
            ("008_realdiv_4096_1024", realdiv_run, ((4096, 1024), (4096, 1024), "float16", "cce_realdiv_fp16"), ),
            ("009_realdiv_1024_4096", realdiv_run, ((1024, 4096), (1024, 4096), "float16", "cce_realdiv_fp16"), ),
        ]

        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
