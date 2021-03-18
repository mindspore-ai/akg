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
from tests.common.test_run.resize_bilinear_grad_run import resize_bilinear_grad_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_resize_bilinear_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("========================{0}  Setup case=================".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            ("resize_bilinear_grad_00", resize_bilinear_grad_run, ([1, 16, 16, 16], [1, 8, 8, 16], "float16", "cce_resize_bilinear_grad_fp16")),
            #("resize_bilinear_grad_01", resize_bilinear_grad_run, ([4,33,33,256],[4,1,1,256],     "float16", "cce_resize_bilinear_grad_fp16")),
            #("resize_bilinear_grad_02", resize_bilinear_grad_run, ([4,129,129,48],[4,129,129,48], "float16", "cce_resize_bilinear_grad_fp16")),
            #("resize_bilinear_grad_03", resize_bilinear_grad_run, ([4,129,129,256],[4,33,33,256], "float16", "cce_resize_bilinear_grad_fp16")),
            #("resize_bilinear_grad_04", resize_bilinear_grad_run, ([4,513,513,21],[4,129,129,21], "float16", "cce_resize_bilinear_grad_fp16")),
            #("resize_bilinear_grad_05", resize_bilinear_grad_run, ([4,129,129,21],[4,129,129,21], "float16", "cce_resize_bilinear_grad_fp16")),
        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
          clean environment
          :return:
          """
        self._log.info("========================{0} Teardown case=================".format(self.casename))
