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
avgpool_grad test cast
"""

import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.avgpool_grad_run import avgpool_grad_run


class TestAvgPoolGrad(TestBase):

    def setup(self):
        case_name = "test_akg_avgpool_grad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg_ci = [
            # testflag,opfuncname,testRunArgs:shape,kernel,stride,pad,dtype, dimArgs
            ("avgpool_grad_01", avgpool_grad_run, ((1, 1, 16, 16, 16), (2, 2), (1, 1), (0, 0, 0, 0), "float16")),
            ("avgpool_grad_01", avgpool_grad_run, ((1, 1, 16, 16, 16), (2, 2), (1, 1), 'VALID', "float16")),
            ("avgpool_grad_01", avgpool_grad_run, ((1, 1, 16, 16, 16), (2, 2), (1, 1), 'SAME', "float16")),
            ("avgpool_grad_02", avgpool_grad_run, ((1, 1, 16, 16, 16), (2, 2), (4, 4), (1, 1, 1, 1), "float16")),
            ("avgpool_grad_03", avgpool_grad_run, ((1, 1, 16, 16, 16), (2, 2), (4, 4), (0, 0, 0, 0), "float16")),
            ("avgpool_grad_03", avgpool_grad_run, ((1, 1, 16, 16, 16), (2, 2), (4, 4), 'VALID', "float16")),
            ("avgpool_grad_03", avgpool_grad_run, ((1, 1, 16, 16, 16), (2, 2), (4, 4), 'SAME', "float16")),
            ("avgpool_grad_04", avgpool_grad_run, ((1, 1, 16, 16, 16), (4, 4), (3, 3), (0, 0, 0, 0), "float16")),
            ("avgpool_grad_04", avgpool_grad_run, ((1, 1, 16, 16, 16), (4, 4), (3, 3), 'VALID', "float16")),
            ("avgpool_grad_04", avgpool_grad_run, ((1, 1, 16, 16, 16), (4, 4), (3, 3), 'SAME', "float16")),
            ("avgpool_grad_05", avgpool_grad_run, ((10, 3, 16, 16, 16), (4, 4), (3, 3), (0, 0, 0, 0), "float16")),
            ("avgpool_grad_05", avgpool_grad_run, ((10, 3, 16, 16, 16), (4, 4), (3, 3), 'VALID', "float16")),
            ("avgpool_grad_05", avgpool_grad_run, ((10, 3, 16, 16, 16), (4, 4), (3, 3), 'SAME', "float16")),
            ("avgpool_grad_06", avgpool_grad_run, ((1, 3, 64, 64, 16), (4, 4), (3, 3), (0, 0, 0, 0), "float16")),
            ("avgpool_grad_06", avgpool_grad_run, ((1, 3, 64, 64, 16), (4, 4), (3, 3), 'VALID', "float16")),
            #  ("avgpool_grad_06", avgpool_grad_run, (( 1,3, 64, 64,16),(  4,  4),(3, 3),      'SAME', "float16")),
            ("avgpool_grad_07", avgpool_grad_run, ((1, 8, 100, 200, 16), (100, 200), (1, 1), (0, 0, 0, 0), "float16")),
            ("avgpool_grad_07", avgpool_grad_run, ((1, 8, 100, 200, 16), (100, 200), (1, 1), 'VALID', "float16")),
            #  ("avgpool_grad_07", avgpool_grad_run, (( 1,8,100,200,16),(100,200),(1, 1),      'SAME', "float16")),
        ]
        self.testarg_debug = [
            # ("avgpool_grad_01", avgpool_grad_run, ((1,1,16,16,16),(2,2),(1,1),(0,0), "float16")),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level3
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_ci(self):
        self.common_run(self.testarg_ci)

    @pytest.mark.level2
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_debug(self):
        self.common_run(self.testarg_debug)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
