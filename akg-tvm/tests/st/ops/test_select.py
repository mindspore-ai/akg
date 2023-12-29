# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run import select_run

class TestCase(TestBase):
    def setup(self):
        case_name = "test_select_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """set test case """
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("select_001", select_run, ((3, ), (2, 3, 3),  "int8", "int8")),
            ("select_001", select_run, ((2, ), (2, ),  "int8", "int32")),
            ("select_001", select_run, ((2, ), (2, ),  "int8", "uint8")),
            ("select_001", select_run, ((2, ), (2, 4, 2),  "int8", "int8")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int8", "int32")),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("select_001", select_run, ((2, ), (2, ),  "int8", "int8")),
            ("select_001", select_run, ((2, ), (2, ),  "int8", "int32")),
            ("select_001", select_run, ((2, ), (2, ),  "int8", "uint8")),
            ("select_001", select_run, ((2, ), (2, ),  "int8", "float16")),
            ("select_001", select_run, ((2, ), (2, ),  "int8", "float32")),
            ("select_001", select_run, ((2, ), (2, ),  "int32", "int8")),
            ("select_001", select_run, ((2, ), (2, ),  "int32", "int32")),
            ("select_001", select_run, ((2, ), (2, ),  "int32", "uint8")),
            ("select_001", select_run, ((2, ), (2, ),  "int32", "float16")),
            ("select_001", select_run, ((2, ), (2, ),  "int32", "float32")),

            ("select_001", select_run, ((2, ), (2, 2, 2),  "int8", "int8")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int8", "int32")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int8", "uint8")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int8", "float16")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int8", "float32")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int32", "int8")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int32", "int32")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int32", "uint8")),
            ("select_001", select_run, ((2, ), (2, 2, 2),  "int32", "float16")),
            ("select_001", select_run, ((2, ), (2, 4, 2),  "int32", "float32")),
        ]
        self.args_default = [
            ("000_case", select_run, ((2, ), (2, 2, 2),  "int8", "float16"), ["level0"]),
            ("001_case", select_run, ((2, ), (2, 2, 2),  "int8", "float32"), ["level0"])
        ]
        
        return True

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.args_default, utils.CUDA, "level0")
    
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cpu_level0(self):
        return self.run_cases(self.args_default, utils.LLVM, "level0")

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
