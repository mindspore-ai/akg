# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.common.test_run import equal_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_auto_equal_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            #caseflag,testfuncname,testRunArgs, dimArgs
            ("001_equal", equal_run, (((128,), (128,)), "float16"), ["level0"]),
            ("002_equal", equal_run, (((128, 128), (128, 128)), "float16"), ["level0"]),
            ("003_equal", equal_run, (((1,), (1,)), "float16"), ["level0"]),
            ("004_equal", equal_run, (((1052676,), (1052676,)), "float16"), ["level0"]),
            ("005_equal", equal_run, (((263169,), (1,)), "int32"), ["level0"]),
            ("006_equal", equal_run, (((1,), (1,)), "int32"), ["level0"]),
            ("007_equal", equal_run, (((1,), (1,)), "float32"), ["level0"]),
            ("008_equal", equal_run, (((1052676,), (1,)), "float32"), ["level0"]),
        ]
        self.args_default = [
            ("000_case", equal_run, (((1, 1024), (1, 1024)), 'float32'), ["level0"]),
            ("001_case", equal_run, (((1, 1024), (1, 1024)), 'float16'), ["level0"]),
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

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        return self.run_cases(self.args_default, utils.CCE, "level0")

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
