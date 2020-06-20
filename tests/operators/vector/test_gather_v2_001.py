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
from base import TestBase
from nose.plugins.attrib import attr
from test_run.gather_v2_run import gather_v2_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_gather_v2_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info(
            "============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ("gather_v2_001", gather_v2_run, ((10,), "int8", (1,), "int32", 0)),
            ("gather_v2_001", gather_v2_run, ((16, ), "float16", (1,),  "int32", 0)),
            ("gather_v2_003", gather_v2_run, ((1001,), "int32", (1,), "int32", 0)),
            ("gather_v2_004", gather_v2_run,
             ((1008,), "float32", (1,), "int32", 0)),
            ('gather_v2_005', gather_v2_run, ((64, 32, 16), 'float16', (6,), 'int32', 1)),
            # ('gather_v2-dal-v1', gather_v2_run, ((227, 28), 'float16', (28, ), 'int32', -1))
            # ('test_op_build-gather_v2-base_func_fp16_0003.param', 'gather_v2_run', ((812, 951, 286),
            # 'float16', (212,), 'int32', -1)),
            # ('test_op_build-gather_v2-dal-v1', gather_v2_run, ([897, 4, 1366, 33], 'float16', (851, ), 'int32', 2))
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info(
            "============= {0} Teardown============".format(self.casename))
        return
