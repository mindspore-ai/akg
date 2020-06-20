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
import pytest
from test_run.pad_run import pad_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_pad"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("pad_01", pad_run, ((128, 1024), ([0, 384], [0, 0]), "float16", "constant", "cce_pad_fp16"), ((128, 1), (128, 128))),
            ("pad_02", pad_run, ((16, 16), ([4, 4], [4, 4]), "int32", "constant", "cce_pad_int32"), ((64, 1), (128, 1))),
            ("pad_03", pad_run, ((1, 341, 500, 3), ([0, 0], [0, 132], [0, 13], [0, 0]), "int32", "constant", "cce_pad_int32"), ),
            ("pad_04", pad_run, ((1, 122), ([0, 0], [0, 6]), "float16", "constant", "cce_pad_float16"), ),

            # Matmul shape
            # ("pad_05", pad_run, ((16384, 33), (), "float32", "constant", "cce_pad_float16"), ),
            ("pad_06", pad_run, ((16384, 33), (), "float16", "constant", "cce_pad_float16"), ),
            # ("pad_07", pad_run, ((33, 16384), (), "float32", "constant", "cce_pad_float16"), ),
            ("pad_08", pad_run, ((1024, 8), (), "float32", "constant", "cce_pad_float16"), ),
            ("pad_09", pad_run, ((1024, 2), (), "float32", "constant", "cce_pad_float16"), ),
            ("pad_10", pad_run, ((2, 1024), (), "float32", "constant", "cce_pad_float16"), ),
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("pad_01", pad_run, ((128, 1024), ([0, 384], [0, 0]), "float32", "constant", "cce_pad_fp32"), ((1, 1), (1024, 1024))),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

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
