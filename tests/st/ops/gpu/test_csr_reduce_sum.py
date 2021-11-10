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
from tests.common.test_run.gpu import csr_reduce_sum_run

############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_abs_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.test_args = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("000_case", csr_reduce_sum_run, ((1987, 800), 'float32', 'int32', -1), ["level0"]),
            ("001_case", csr_reduce_sum_run, ((10000, 10000), 'float32', 'int32', -1), ["level0"]),
        ]
        return True

    def teardown(self):
        self._log.info("{0} Teardown".format(self.casename))
        super(TestCase, self).teardown()
        return

    @pytest.mark.level0
    @pytest.mark.platform_x86_gpu_training
    @pytest.mark.env_onecard
    def test_gpu_level0(self):
        return self.run_cases(self.test_args, utils.CUDA, "level0")