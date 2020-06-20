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
matmul4d_ad
"""
import datetime
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.matmul4d_ad_run import matmul4d_ad_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_matmul_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag, opfuncname, testRunArgs, dimArgs
            # shape_x, shape_y, bias, bypass, adj_x, adj_y, dtype, out_dtype, kernel_name, attrs
            ("matmul4d_ad_run_0", matmul4d_ad_run, ((64, 128), (128, 32), 0,   False, False,
                                                    "float16", "float16", "matmul4d_ad_cce")),
            ("matmul4d_ad_run_1", matmul4d_ad_run, ((64, 1024), (1024, 32), 0,   False, False,
                                                    "float16", "float16", "matmul4d_ad_cce")),
            ("matmul4d_ad_run_2", matmul4d_ad_run, ((1024, 64), (1024, 32), 0,  True, False,
                                                    "float16", "float16", "matmul4d_ad_cce")),
            ("matmul4d_ad_run_3", matmul4d_ad_run, ((64, 1024), (32, 1024), 0,   False, True,
                                                    "float16", "float16", "matmul4d_ad_cce")),
            ("matmul4d_ad_run_4", matmul4d_ad_run, ((1024, 64), (32, 1024), 0, True, True,
                                                    "float16", "float16", "matmul4d_ad_cce")),
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
