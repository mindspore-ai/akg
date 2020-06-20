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
from base import TestBase
from nose.plugins.attrib import attr


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):

    def setup(self):
        case_name = "FusedLayerNormTest_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #  ("fused_layernorm_01", "fused_layernorm_run", ((1280, 1024), 1, -1, 'float16'), ((1,1),(3070,3070))),
            #  ("fused_layernorm_02", "fused_layernorm_run", ((32, 64, 512), 1, -1, 'float16')), # setdim wrong
            #  ("fused_layernorm_02", "fused_layernorm_run", ((16, 16, 16), 1, -1, 'float16')),
            #  ("fused_layernorm_02", "fused_layernorm_run", ((64, 128, 1024), 1, -1, 'float16')), # precision problem
            ("fused_layernorm_03", "fused_layernorm_run", ((16, 16, 1, 1), 1, 1, 'float16')),
            #  ("fused_layernorm_03", "fused_layernorm_run", ((2048, 512), 1, -1, 'float16')),
            #  ("fused_layernorm_04", "fused_layernorm_run", ((2964, 256), 1, -1, 'float16')),
            #  ("fused_layernorm_05", "fused_layernorm_run", ((32, 128, 256), 2, -1, 'float16')),
            #  ("fused_layernorm_06", "fused_layernorm_run", ((8192, 256), 1, -1, 'float16')),
            #  ("fused_layernorm_06", "fused_layernorm_run", ((8192, 1024), 1, -1, 'float16'), ((1,1),(3070,3070))),
        ]

        self.testarg_rpc_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            #  ("fused_layernorm_03", "fused_layernorm_run", ((2048, 512), 1, -1, 'float32')),
            # float16:[64 * 128, 1024] = float16:[64 * 128, 1024]
            ("fused_layernorm_001_8192_1024", "fused_layernorm_run", ((64 * 128, 1024), 1, -1, 'float16')),
            # float16:[64 * 20, 1024] = float:[64 * 20, 1024]
            ("fused_layernorm_002_1280_1024", "fused_layernorm_run", ((64 * 20, 1024), 1, -1, 'float16')),
            ("fused_layernorm_1_768", "fused_layernorm_run", ((1, 768), 1, 1, 'float32')),
        ]
        self.testarg_level1 = [
            #  ("fused_layernorm_01", "fused_layernorm_run", ((1280, 1024), 1, -1, 'float16')),
            #  ("fused_layernorm_02", "fused_layernorm_run", ((32, 64, 512), 1, -1, 'float16')),
            #("fused_layernorm_03", "fused_layernorm_run", ((2048, 512), 1, -1, 'float16')),
            #("fused_layernorm_04", "fused_layernorm_run", ((2964, 256), 1, -1, 'float16')),
            #  ("fused_layernorm_05", "fused_layernorm_run", ((32, 128, 256), 2, -1, 'float16')),
            #  ("fused_layernorm_06", "fused_layernorm_run", ((8192, 256), 1, -1, 'float16')),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run([self.testarg_rpc_cloud[1]])

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
