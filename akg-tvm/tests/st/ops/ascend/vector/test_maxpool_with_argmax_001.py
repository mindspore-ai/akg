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

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run.ascend.maxpool_with_argmax_run import maxpool_with_argmax_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_maxpool_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs:shape,kernel,stride,pad,hybrid,dtype,dimArgs
            # reid
            # others

            # not hybrid

            # not hybrid manual schedule
        ]
        self.testarg_l1 = [
        ]
        self.testarg_aic = [
        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs:shape,kernel,stride,pad,dtype, dimArgs
            # resnet 50
            ("resnet50_maxpool_fp16_c", maxpool_with_argmax_run, ((32, 4, 112, 112, 16), (3, 3), (2, 2), (1, 1, 1, 1), True, "float16")),
#            ("resnet50_maxpool_fp16_t", maxpool_with_argmax_run, ((32, 4, 112, 112, 16), (3, 3), (2, 2), (0, 1, 0, 1), True, "float16")),
#            ("resnet50_maxpool_fp16_t2", maxpool_with_argmax_run, ((32, 4, 112, 112, 16), (3, 3), (2, 2), 'SAME', True, "float16")),
        ]
        self.testarg_level2 = [
            # testflag,opfuncname,testRunArgs:shape,kernel,stride,pad,dtype, dimArgs
            # ("005_maxpool", maxpool_run, ((1,1,16,16,16),(2,2),(2,2),(3,3,3,3), True,   "float16")), # Variable not found in isl
            # ("004_maxpool", maxpool_run, ((1,1,16,16,16),(2,2),(3,3),(2,2,2,2), True,  "float16")), # false Variable not found in isl
            # ("002_maxpool", maxpool_run, ((1,1,16,16,16),(2,2),(2,2),(1,1,1,1), True,  "float32")), # terminate called after throwing an instance of 'std::logic_error'
            # ("011_maxpool", maxpool_run, ((1,1,30,30,16),(2,2),(1,1),(2,2,2,2), True,  "float16")), # terminate called after throwing an instance of 'std::logic_error'

        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_l1(self):
        self.common_run(self.testarg_l1 + self.testarg_cloud)

    def test_run_aic(self):
        self.common_run(self.testarg_aic)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_level2(self):
        self.common_run(self.testarg_level2)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
