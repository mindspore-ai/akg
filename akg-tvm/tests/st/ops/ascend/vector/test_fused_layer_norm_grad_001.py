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

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestFusedLayerNormGrad(TestBase):

    def setup(self):
        case_name = "test_akg_fused_layer_norm_grad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            # ("fused_layer_norm_grad_0","fused_layer_norm_grad_run",((128, 256), 1, -1, "float16")),
            ("fused_layer_norm_grad_01", "fused_layer_norm_grad_run", ((16, 16), 1, -1, "float16")),
            # assertion error
            #("fused_layer_norm_grad_02", "fused_layer_norm_grad_run", ((128, 1024), -1, -1, "float16")),

            #  ("fused_layer_norm_grad_01","fused_layer_norm_grad_run",((16, 16, 16), 1, 1, "float16")),
            # ("fused_layer_norm_grad_02","fused_layer_norm_grad_run",((128, 256), 1, -1, "float16")),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, dimArgs
            #  ("fused_layer_norm_grad_01","fused_layer_norm_grad_run",((128, 256), 1, -1, "float32")),
        ]
        self.testarg_rpc_cloud = [
            ("fused_layer_norm_grad_0", "fused_layer_norm_grad_run", ((16, 16), 1, -1, "float16")),
            ("fused_layer_norm_grad_03", "fused_layer_norm_grad_run", ([1, 128, 1024], -1, -1, "float32")),
            ("fused_layer_norm_grad_1", "fused_layer_norm_grad_run", ([256, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_2", "fused_layer_norm_grad_run", ([320, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_3", "fused_layer_norm_grad_run", ([640, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_4", "fused_layer_norm_grad_run", ([512, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_5", "fused_layer_norm_grad_run", ([1024, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_6", "fused_layer_norm_grad_run", ([2560, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_7", "fused_layer_norm_grad_run", ([128, 1024], 1, 1, "float32")),
            ("fused_layer_norm_grad_8", "fused_layer_norm_grad_run", ([256, 1024], 1, 1, "float32")),
            ("fused_layer_norm_grad_9", "fused_layer_norm_grad_run", ([1, 128, 768], 2, 2, "float32")),
            ("fused_layer_norm_grad_10", "fused_layer_norm_grad_run", ([128, 128, 768], 2, 2, "float32")),
            ("fused_layer_norm_grad_11", "fused_layer_norm_grad_run", ([8, 128, 768], 2, 2, "float32")),
            ("fused_layer_norm_grad_12", "fused_layer_norm_grad_run", ([8192, 768], -1, -1, "float32")),
            ("fused_layer_norm_grad_13", "fused_layer_norm_grad_run", ([8192, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_15", "fused_layer_norm_grad_run", ([15360, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_16", "fused_layer_norm_grad_run", ([16, 128, 768], 2, 2, "float32")),  # issue 794
            ("fused_layer_norm_grad_17", "fused_layer_norm_grad_run", ([2048, 768], 1, 1, "float32")),
            ("fused_layer_norm_grad_18", "fused_layer_norm_grad_run", ([160, 1024], 1, 1, "float32")),
            # float16:[64 * 128, 1024] = float16:[64 * 128, 1024]
            ("fused_layernorm_001_8192_1024", "fused_layer_norm_grad_run", ((64 * 128, 1024), 1, -1, 'float16')),
        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
