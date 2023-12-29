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
from tests.common.test_run import reduce_prod_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_reduce_prod"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("reduce_prod_01", reduce_prod_run, ((1195,), "float32", (0,), True)),
        ]

        self.test_args = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("000_case", reduce_prod_run, ((32,), 'float32', None, True), ["level0"]),
            ("001_case", reduce_prod_run, ((65536, 3), 'float32', (1,), True), ["level0"]),
            ("002_case", reduce_prod_run, ((256, 32, 1024), 'float32', (1,), False), ["level0"]),
            ("003_case", reduce_prod_run, ((1195,), "float16", None, True)),
            ("004_case", reduce_prod_run, ((1195,), "float32", (0,), True)),
            ("005_case", reduce_prod_run, ((1195,), "int8", (0,), True)),
            ("006_case", reduce_prod_run, ((1195,), "uint8", (0,), True)),
        ]
        return True

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_ascend_run(self):
        return self.run_cases(self.test_args, utils.CCE, "level0")

    def test_ascend_cloud_run(self):
        return self.run_cases(self.testarg_cloud, utils.CCE, "level0")

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.test_args, utils.CUDA, "level0")
    
    @pytest.mark.level0
    @pytest.mark.platform_x86_cpu
    @pytest.mark.env_onecard
    def test_cpu_level0(self):
        return self.run_cases(self.test_args, utils.LLVM, "level0")