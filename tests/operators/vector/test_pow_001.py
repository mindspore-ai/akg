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
tf pow
"""
import os
from base import TestBase
import pytest


############################################################
# TestCase= class: put to tests/*/
############################################################
class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_pow_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            ("test_1024_4096", "pow_run", ((1024, 4096), (1024, 4096), 'float16'),),
            ("test_1024_4096", "pow_run", ((5, 5, 4, 4), (5, 5, 4, 4), 'int32'),),
            ("test_16", "pow_run", ((16, ), (2, 16), 'int8'),),
            ("test_16_2", "pow_run", ((2, 16), (16,), 'uint8'),),
            # test for broadcast
            ("test_1024_4096", "pow_run", ((1024, 4096), (1, 4096), 'float16'),),
            ("test_1024_4096", "pow_run", ((1, 4096), (1024, 4096), 'float16'),),
            ("test_1024_4096", "pow_run", ((1024, 4096), (1024, 4096), 'float16'),),
            ##("test_8192_4096", "pow_run", ((8192,4096), 'float16'), ((4,4), (4096,4096))),
            ("test_1280_1024", "pow_run", ((1280, 1024), (1280, 1024), 'float16'),),
            ("test_160_1024", "pow_run", ((160, 1024), (160, 1024), 'float16'),),
            # add rpc testcase
            ("test_1_128", "pow_run", ((1, 128), (1, 128), 'float16'),),
            ("test_128_128", "pow_run", ((128, 128), (128, 128), 'float16'),),
            ("test_128_256", "pow_run", ((128, 256), (128, 256), 'float16'),),
        ]
        self.testarg_rpc_cloud = [
            ("test_1024_4096", "pow_run", ((1024, 4096), (1024, 4096), 'float16'),),
        ]
        self.testarg_cloud = [
            ("test_160_1024", "pow_run", ((160, 1024), (160, 1024), 'float32'),),
        ]
        self.testarg_level1 = [
            ## testflag,opfuncname,testRunArgs, dimArgs
            # ("test_1024_4096", "pow_run", ((1024,4096), 'float16'), ((4,4), (4096,4096))),
            ("test_8192_4096", "pow_run", ((8192, 4096), (8192, 4096), 'float16'),)
            # ("test_8192_4096", "pow_run", ((8192,4096), 'float16'), ((4,4), (4096,4096))),

            # ("test_1280_1024", "pow_run", ((1280,1024), 'float16'), ((16,16), (1024,1024))),
            # ("test_160_1024",  "pow_run", ((160,1024), 'float16'),  ((16,16), (1024,1024))),
            # add rpc testcase
            # ("test_1_128", "pow_run", ((1, 128), 'float16'), ((128, 128), (128, 128))),
            # ("test_128_128", "pow_run", ((128, 128), 'float16'), ((0, 0), (128, 128))),
            # ("test_128_256", "pow_run", ((128, 256), 'float16'), ((0, 0), (128, 128))),
        ]
        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    @pytest.mark.aicmodel
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_cloud)

    @pytest.mark.level1
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_level1(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_level1)

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
