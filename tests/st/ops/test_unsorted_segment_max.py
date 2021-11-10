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

"""testcase for unsorted_segment_max op"""

import os
import pytest
from tests.common.base import TestBase
from tests.common.test_run import unsorted_segment_max_run


class TestCase(TestBase):
    def setup(self):
        case_name = "test_akg_unsorted_segment_max_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)

        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.args = [
            # testflag, opfuncname, testRunArgs, dimArgs
            #("001_fp16",  unsorted_segment_max_run, ([128,32], [128],12,"float16")),
            #("002_fp16",  unsorted_segment_max_run, ([512,256], [512],64,  "float16")),
            #("003_fp16",  unsorted_segment_max_run, ([128,16,16], [128],10, "float16")),
            #("004_fp16",  unsorted_segment_max_run, ([1024,1024], [1024],128, "float16")),
            #("005_fp32",  unsorted_segment_max_run, ([256,1024], [256],32, "float32")),
            #("006_fp16",  unsorted_segment_max_run, ([128, 64,32], [128],9, "float16")),
            # fail
            #("006_fp16",  segment_max_run, ([128, 64,32], [128], "float16"),((128,128),(16,16),(32,32))),
            #("006_fp16",  segment_max_run, ([128, 64,32], [128], "float16"),((128,128),(64,64),(16,16))),
        ]

        return True

    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend_training
    @pytest.mark.platform_x86_ascend_training
    @pytest.mark.env_onecard
    def test_run(self):
        """
        run case.#
        :return:
        """
        '''
        from print_debug import print_debug
        for arg in self.args:
          caseflag, func, args = self.ana_args(arg)
          print_debug(self,caseflag,arg,True) '''
        self.common_run(self.args)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
        return


if __name__ == "__main__":
    t = TestCase()
    t.setup()
    t.test_run()
    t.teardown()
