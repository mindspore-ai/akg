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

import os
import pytest
from tests.common.base import get_splitted_cases
from tests.st.composite.test_composite_json import test_single_file


def test_network(info_dir, split_nums=1, split_idx=0):
    pwd = os.path.dirname(os.path.abspath(__file__))
    all_files = os.listdir(info_dir)
    files = get_splitted_cases(all_files, split_nums, split_idx)
    for file in files:
        poly = True
        attrs = None
        file_path = pwd + "/" + info_dir + "/" + file
        test_single_file(file_path, attrs, poly, profiling=False)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_adam_gpu_level0():
    test_network("./gpu/bert_adam/level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_adam_gpu_level1():
    test_network("./gpu/bert_adam/level1")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deep_fm_gpu_level0():
    test_network("./gpu/deep_fm/level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deep_fm_gpu_level1():
    test_network("./gpu/deep_fm/level1")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_gpu_level0():
    test_network("./gpu/wide_deep/level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_gpu_level1():
    test_network("./gpu/wide_deep/level1")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_ps_gpu_level0():
    test_network("./gpu/wide_deep_ps/level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_ps_gpu_level1():
    test_network("./gpu/wide_deep_ps/level1")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level0_test0():
    test_network("./gpu/yolov3_darknet53/level0", 2, 0)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level0_test1():
    test_network("./gpu/yolov3_darknet53/level0", 2, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level1_test0():
    test_network("./gpu/yolov3_darknet53/level1", 2, 0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level1_test1():
    test_network("./gpu/yolov3_darknet53/level1", 2, 1)
