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

"""
batch_norm_ad
"""

import os
import pytest
from tests.common.base import TestBase, get_splitted_cases
from tests.common.test_run.batch_norm_ad_run import batch_norm_ad_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_batch_norm_ad_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            ##Resnet50 shapes
            ("batch_norm_ad_run5", batch_norm_ad_run, ((32, 4, 112, 112, 16), "float32", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run9", batch_norm_ad_run, ((32, 8, 56, 56, 16), "float32", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run4", batch_norm_ad_run,((32, 32, 7, 7, 16),"float32", 1e-5,"batch_norm_ad_output")),
            ("batch_norm_ad_run11", batch_norm_ad_run, ((32, 32, 14, 14, 16),"float32", 1e-5,"batch_norm_ad_output")),
            # output is nan
            #("batch_norm_ad_run9", batch_norm_ad_run, ((256, 8, 56, 56, 16), "float32", 1e-5, "batch_norm_ad_output")),
            #  Need new dims ("batch_norm_ad_run5", batch_norm_ad_run, ((32, 4, 112, 112, 16), "float16", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run9", batch_norm_ad_run, ((32, 8, 56, 56, 16), "float16", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run4", batch_norm_ad_run,((32, 32, 7, 7, 16),"float16", 1e-5,"batch_norm_ad_output")),
            ("batch_norm_ad_run11", batch_norm_ad_run, ((32, 32, 14, 14, 16),"float16", 1e-5,"batch_norm_ad_output")),
        ]
        self.testarg_cloud = [
            ###Resnet50 shapes
            ## FLOAT32 tests that passed TOL = 1e-4
            ("batch_norm_ad_run0", batch_norm_ad_run,((32, 128, 7, 7, 16),"float32", 1e-5,"batch_norm_ad_output")),
            ("batch_norm_ad_run1", batch_norm_ad_run, ((32, 16, 14, 14, 16),"float32", 1e-6,"batch_norm_ad_output")),
            ("batch_norm_ad_run2", batch_norm_ad_run, ((32, 16, 56, 56, 16), "float32", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run3",batch_norm_ad_run,((32, 32, 28, 28, 16),"float32", 1e-5,"batch_norm_ad_output")),
            ("batch_norm_ad_run4", batch_norm_ad_run,((32, 32, 7, 7, 16),"float32", 1e-5,"batch_norm_ad_output")),
            ("batch_norm_ad_run5", batch_norm_ad_run, ((32, 4, 112, 112, 16), "float32", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run6", batch_norm_ad_run, ((32, 4, 56, 56, 16), "float32", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run7", batch_norm_ad_run,((32, 64, 14, 14, 16),"float32", 1e-5,"batch_norm_ad_output")),
            ("batch_norm_ad_run8", batch_norm_ad_run, ((32, 8, 28, 28, 16), "float32", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run9", batch_norm_ad_run, ((32, 8, 56, 56, 16), "float32",1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run10", batch_norm_ad_run, ((32, 16, 28, 28, 16), "float32", 1e-5, "batch_norm_ad_output")),
            ("batch_norm_ad_run11", batch_norm_ad_run, ((32, 32, 14, 14, 16),"float32", 1e-5,"batch_norm_ad_output")),

            ## FLOAT16, all passed TOL = 1e-3
            # time 0.19ms -- 0.15ms
            ("batch_norm_ad_run0", batch_norm_ad_run,((32, 128, 7, 7, 16),"float16", 1e-5,"batch_norm_ad_output"),),
            # time 0.16ms -- 0.13ms
            ("batch_norm_ad_run1", batch_norm_ad_run, ((32, 16, 14, 14, 16),"float16", 1e-5,"batch_norm_ad_output")),
            # time 2.37ms -- 1.7ms
            ("batch_norm_ad_run2", batch_norm_ad_run, ((32, 16, 56, 56, 16), "float16", 1e-5, "batch_norm_ad_output")),
            # time 0.61ms -- 0.5ms
            ("batch_norm_ad_run3", batch_norm_ad_run,((32, 32, 28, 28, 16),"float16", 1e-5,"batch_norm_ad_output")),
            # time 0.05ms -- 0.04ms
            ("batch_norm_ad_run4", batch_norm_ad_run,((32, 32, 7, 7, 16),"float16", 1e-5,"batch_norm_ad_output"),),
            # time 9.37ms -- 6.67ms
            # Need new dims ("batch_norm_ad_run5", batch_norm_ad_run, ((32, 4, 112, 112, 16), "float16", 1e-5, "batch_norm_ad_output")),
            # time 2.35ms -- 1.6ms
            ("batch_norm_ad_run6", batch_norm_ad_run, ((32, 4, 56, 56, 16), "float16", 1e-5, "batch_norm_ad_output")),
            # time 0.38ms -- 2.6ms
            ("batch_norm_ad_run7", batch_norm_ad_run,((32, 64, 14, 14, 16),"float16", 1e-5,"batch_norm_ad_output")),
            # time 0.59ms -- 0.45ms
            ("batch_norm_ad_run8", batch_norm_ad_run, ((32, 8, 28, 28, 16), "float16", 1e-5, "batch_norm_ad_output")),
            # time 2.36ms -- 1.6ms
            ("batch_norm_ad_run9", batch_norm_ad_run, ((32, 8, 56, 56, 16), "float16", 1e-5, "batch_norm_ad_output")),
            # time 0.6ms -- 0.42ms
            ("batch_norm_ad_run10", batch_norm_ad_run, ((32, 16, 28, 28, 16), "float16", 1e-5, "batch_norm_ad_output")),
            # time 0.17ms -- 0.13ms
            ("batch_norm_ad_run11", batch_norm_ad_run, ((32, 32, 14, 14, 16),"float16", 1e-5,"batch_norm_ad_output")),
        ]
        return

    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    def test(self, split_nums, split_idx):
        self.common_run(get_splitted_cases(self.testarg, split_nums, split_idx))

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test0():
    a = TestCase()
    a.setup()
    a.test(3, 0)
    a.teardown()


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test1():
    a = TestCase()
    a.setup()
    a.test(3, 1)
    a.teardown()


@pytest.mark.level2
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test2():
    a = TestCase()
    a.setup()
    a.test(3, 2)
    a.teardown()
