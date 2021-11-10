# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.common.test_run.ascend.sigmoid_cross_entropy_with_logits_grad_run import sigmoid_cross_entropy_with_logits_grad_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_sigmoid_cross_entropy_with_logits_grad_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        """set test case """
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("sigmoid_cross_entropy_with_logits_grad_001", sigmoid_cross_entropy_with_logits_grad_run,
             ((16,), "float16")),
            ("sigmoid_cross_entropy_with_logits_grad_004", sigmoid_cross_entropy_with_logits_grad_run,
             ((224, 224), "float32")),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("sigmoid_cross_entropy_with_logits_grad_001", sigmoid_cross_entropy_with_logits_grad_run,
             ((16,), "float16")),
            ("sigmoid_cross_entropy_with_logits_grad_002", sigmoid_cross_entropy_with_logits_grad_run,
             ((224,), "float16")),
            ("sigmoid_cross_entropy_with_logits_grad_003", sigmoid_cross_entropy_with_logits_grad_run,
             ((16, 16), "float32")),
            ("sigmoid_cross_entropy_with_logits_grad_004", sigmoid_cross_entropy_with_logits_grad_run,
             ((224, 224), "float32")),
        ]

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
