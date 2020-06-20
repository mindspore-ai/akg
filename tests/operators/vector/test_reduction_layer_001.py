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

"""reduction_layer testcases"""

import os
from base import TestBase
import pytest


class TestCase(TestBase):
    def setup(self):
        case_name = "test_reduction_layer_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            # shape, dtype, axis, op, coeff
            ("reduction_layer_001", "reduction_layer_run", ((16, 16), "float16", 0, "SUM", 0.1)),
            ("reduction_layer_002", "reduction_layer_run", ((1, 1, 16, 16), "float32", -3, "ASUM", 0.2)),
            ("reduction_layer_003", "reduction_layer_run", ((1, 1, 1, 16), "float32", -3, "SUMSQ", 0.2)),
            ("reduction_layer_004", "reduction_layer_run", ((1, 1, 1, 16), "float32", -3, "MEAN", 1)),
            ("reduction_layer_005", "reduction_layer_run", ((1, 1, 1, 16), "int8", -3, "SUMSQ", 0.2)),
            ("reduction_layer_006", "reduction_layer_run", ((1, 16), "uint8", 1, "MEAN", 1)),
        ]
        self.testarg_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, axis, op, coeff
        ]

        self.testarg_rpc_cloud = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, axis, op, coeff
        ]
        self.testarg_level1 = [
            # testflag, opfuncname, testRunArgs, setdimArgs
            # shape, dtype, axis, op, coeff
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        self.common_run(self.testarg_level1)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
