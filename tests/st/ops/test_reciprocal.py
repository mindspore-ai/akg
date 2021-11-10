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

"""testcase for reciprocal op"""
import os
import pytest
import akg.utils as utils
from tests.common.base import TestBase
from tests.common.test_run import reciprocal_run

class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_reciprocal_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_1024_4096", reciprocal_run, ((1024, 4096), 'float16'), ["level0"]),
            ("test_1280_1024", reciprocal_run, ((1280, 1024), 'float16'), ["level0"]),
            ("test_160_1024", reciprocal_run, ((160, 1024), 'float16'), ["level0"]),
            ("test_1_128", reciprocal_run, ((1, 128), 'float16'), ["level0"]),
            ("test_128_128", reciprocal_run, ((128, 128), 'float16'), ["level0"]),
            ("test_128_256", reciprocal_run, ((128, 256), 'float16'), ["level0"]),
        ]
        self.testarg_cloud = [
            ("test_160_1024", reciprocal_run, ((160, 1024), 'float32'),),
        ]
        return True

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.testarg, utils.CUDA, "level0")
    
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cpu_level0(self):
        return self.run_cases(self.testarg, utils.LLVM, "level0")

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        return self.run_cases(self.testarg, utils.CCE, "level0")

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
