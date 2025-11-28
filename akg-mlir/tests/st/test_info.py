# Copyright 2025 Huawei Technologies Co., Ltd
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

"""AKG-MLIR st test."""
import os
import pytest

class TestCase:
    """class TestCase."""

    def run_case(self, info, target='ascend'):
        """ run a test case """
        cmd = "akg_benchmark -e " +  target + " -f " + info
        ret = os.system(cmd)
        assert ret == 0


    @pytest.mark.level0
    @pytest.mark.platform_arm_ascend
    @pytest.mark.env_onecard
    def test_fuse_sub_add(self):
        """
        Feature: AKG compile test.
        Description: fuse sub and add.
        Expectation: success
         """
        pwd = os.path.dirname(os.path.abspath(__file__))
        info = os.path.join(pwd, "ascend/fused_sub_add.info")
        return self.run_case(info)
