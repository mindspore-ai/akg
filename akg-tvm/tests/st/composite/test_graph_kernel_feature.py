# Copyright 2022 Huawei Technologies Co., Ltd
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
import subprocess
import pytest
from tests.common.base import get_splitted_cases
from tests.st.composite.test_composite_json import test_single_file


@pytest.mark.skip
def test_feature(dir, level, split_nums=1, split_idx=0):
    pwd = os.path.dirname(os.path.abspath(__file__))
    script_file = os.path.join(pwd, "run_composite_json.py")

    def prepare_script(pwd, script_file):
        if os.path.isfile(script_file):
            return
        src = os.path.join(pwd, "../composite/test_composite_json.py")
        subprocess.call("cp %s %s" % (src, script_file), shell=True)

    prepare_script(pwd, script_file)
    output = os.path.join(pwd, "output" + "_" + dir,  level)
    if not os.path.isdir(output):
        os.makedirs(output, exist_ok=True)
    files_path = os.path.join(pwd, dir, level)
    all_files = os.listdir(files_path)
    all_files.sort()
    files = get_splitted_cases(all_files, split_nums, split_idx)
    for item in files:
        file_path = os.path.join(files_path, item)
        poly = True
        attrs = None
        test_single_file(file_path, attrs, poly, profiling=False)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_elemany_gpu_level0():
    test_feature("elemany", "level0")

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_input_to_attr_ops_gpu_level0():
    test_feature("input_to_attr_ops", "level0")