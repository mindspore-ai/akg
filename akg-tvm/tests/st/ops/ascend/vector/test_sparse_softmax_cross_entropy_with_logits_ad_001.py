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
from tests.common.test_run.ascend.sparse_softmax_cross_entropy_with_logits_ad_run import sparse_softmax_cross_entropy_with_logits_ad_run


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_sparse_softmax_cross_entropy_with_logits_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info(
            "============= {0} Setup case============".format(
                self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("sparse_softmax_cross_entropy_with_logits_ad_02", sparse_softmax_cross_entropy_with_logits_ad_run, [(8,), "int32", (8, 8), "float16", "none", "sparse_softmax_cross_entropy_with_logits_fp16"],),
            ("sparse_softmax_cross_entropy_with_logits_ad_03", sparse_softmax_cross_entropy_with_logits_ad_run, [(16,), "int32", (16, 16), "float16", "sum", "sparse_softmax_cross_entropy_with_logits_fp16"]),
            ("sparse_softmax_cross_entropy_with_logits_ad_04", sparse_softmax_cross_entropy_with_logits_ad_run, [(32,), "int32", (32, 1001), "float16", "mean", "sparse_softmax_cross_entropy_with_logits_fp16", 1024.0/32]),
            ("sparse_softmax_cross_entropy_with_logits_ad_05", sparse_softmax_cross_entropy_with_logits_ad_run, [(1,), "int32", (1, 1001), "float16", "none", "sparse_softmax_cross_entropy_with_logits_fp16"]),
            ("sparse_softmax_cross_entropy_with_logits_ad_06", sparse_softmax_cross_entropy_with_logits_ad_run, [(32,), "int32", (32, 10), "float16", "mean", "sparse_softmax_cross_entropy_with_logits_fp16", 1024.0/32]),
        ]
        self.testarg_cloud = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # resnet50
            ("sparse_softmax_cross_entropy_with_logits_ad_fp32_01", sparse_softmax_cross_entropy_with_logits_ad_run, [(32,), "int32", (32, 1001), "float32", "none", "sparse_softmax_cross_entropy_with_logits_fp32", 1024.0/32],),
            ("sparse_softmax_cross_entropy_with_logits_ad_fp32_02", sparse_softmax_cross_entropy_with_logits_ad_run,  [(32,), "int32", (32, 10), "float32", "none", "sparse_softmax_cross_entropy_with_logits_fp32", 1024.0/32],),

        ]
        return

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.level1
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info(
            "============= {0} Teardown============".format(
                self.casename))
        return


if __name__ == "__main__":
    a = TestCase()
    a.setup()
    a.test_run()
    a.teardown()
