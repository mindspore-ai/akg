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

"""bitwise_and test case"""
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.bitwise_and_run import bitwise_and_run


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_bitwise_and_001"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_bitwise_and", bitwise_and_run, ((12,), "int16", (12,), "int16", "bitwise_and")),
            ("002_bitwise_and", bitwise_and_run, ((12, 16), "int16", (16,), "int16", "bitwise_and")),
            ("003_bitwise_and", bitwise_and_run, ((12,), "uint16", (12,), "uint16", "bitwise_and")),
            ("004_bitwise_and", bitwise_and_run, ((16,), "uint16", (16,), "uint16", "bitwise_and")),
            ("005_bitwise_and", bitwise_and_run, ((12, 16), "uint16", (12, 16), "uint16", "bitwise_and")),
            ("006_bitwise_and", bitwise_and_run, ((5, 5, 5), "uint16", (5, 5, 5), "uint16", "bitwise_and")),
        ]

        self.testarg_cloud = [
            # caseflag,testfuncname,testRunArgs, dimArgs
            ("001_bitwise_and", bitwise_and_run, ((512,), "int16", (512,), "int16", "bitwise_and")),
            ("002_bitwise_and", bitwise_and_run, ((32, 16), "int16", (32, 16), "int16", "bitwise_and")),
            ("003_bitwise_and", bitwise_and_run, ((512,), "uint16", (512,), "uint16", "bitwise_and")),
            ("004_bitwise_and", bitwise_and_run, ((32, 16), "uint16", (32, 16), "uint16", "bitwise_and")),
        ]

        return

    @pytest.mark.rpc_mini
    @pytest.mark.level0
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run(self):
        self.common_run(self.testarg)

    @pytest.mark.rpc_cloud
    @pytest.mark.env_onecard
    @pytest.mark.platform_x86_ascend_training
    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def teardown(self):
        self._log.info("============= {0} Teardown============".format(self.casename))
        return
