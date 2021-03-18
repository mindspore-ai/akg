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

"""testcase for pack"""

import os
from tests.common.base import TestBase
from tests.common.test_run.pack_run import pack_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_pack_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag, opfuncname, testRunArgs, dimArgs
            ('pack_000', pack_run, ([[8, 16], [40, 16]], 'bool', 0)),
            ('pack_001', pack_run, ([[8, 16], [40, 16]], 'int8', 0)),
            ('pack_002', pack_run, ([[8, 16], [40, 16]], 'int16', 0)),
            ('pack_003', pack_run, ([[8, 16], [40, 16]], 'int32', 0)),
            ('pack_004', pack_run, ([[8, 16], [40, 16]], 'int64', 0)),
            ('pack_005', pack_run, ([[8, 16], [40, 16]], 'uint8', 0)),
            ('pack_006', pack_run, ([[8, 16], [40, 16]], 'uint16', 0)),
            ('pack_007', pack_run, ([[8, 16], [40, 16]], 'uint32', 0)),
            ('pack_008', pack_run, ([[8, 16], [40, 16]], 'uint64', 0)),
            ('pack_009', pack_run, ([[8, 16], [40, 16]], 'float16', 0)),
            ('pack_010', pack_run, ([[8, 16], [40, 16]], 'float32', 0)),
        ]
        self.testarg_rpc_cloud = [
            #testflag, opfuncname, testRunArgs, dimArgs
            ('pack_011', pack_run, ([[30522, 1024], [8192, 1024]], 'float16', 0)),
            ('pack_012', pack_run, ([[30522, 1024], [8192, 1024]], 'float32', 0)),
            ('pack_013', pack_run, ([[30522, 1024], [8192, 1024]], 'int8', 0)),
            ('pack_014', pack_run, ([[30522, 1024], [8192, 1024]], 'int16', 0)),
            ('pack_015', pack_run, ([[30522, 1024], [8192, 1024]], 'int32', 0)),
            ('pack_016', pack_run, ([[30522, 1024], [8192, 1024]], 'int64', 0)),
            ('pack_017', pack_run, ([[30522, 1024], [8192, 1024]], 'uint8', 0)),
            ('pack_018', pack_run, ([[30522, 1024], [8192, 1024]], 'uint16', 0)),
            ('pack_019', pack_run, ([[30522, 1024], [8192, 1024]], 'uint32', 0)),
            ('pack_020', pack_run, ([[30522, 1024], [8192, 1024]], 'uint64', 0)),
        ]
        return

    def test_run(self):
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """clean environment"""
        self._log.info("============= {0} Teardown============".format(self.casename))
