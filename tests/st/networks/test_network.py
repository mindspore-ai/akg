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
import subprocess
import pytest
from tests.common.base import get_splitted_cases
from tests.st.composite.test_composite_json import test_single_file
from tests.st.networks.comm_functions import compare_base_line


@pytest.mark.skip
def test_network(backend, network, level, split_nums=1, split_idx=0, check_performance=False):
    pwd = os.path.dirname(os.path.abspath(__file__))
    script_file = os.path.join(pwd, "run_composite_json.py")

    def prepare_script(pwd, script_file):
        if os.path.isfile(script_file):
            return
        src = os.path.join(pwd, "../composite/test_composite_json.py")
        subprocess.call("cp %s %s" % (src, script_file), shell=True)

    prepare_script(pwd, script_file)
    output = os.path.join(pwd, "output" + "_" + backend, network, level)
    if not os.path.isdir(output):
        os.makedirs(output, exist_ok=True)
    files_path = os.path.join(pwd, backend, network, level)
    all_files = os.listdir(files_path)
    files = get_splitted_cases(all_files, split_nums, split_idx)
    for item in files:
        file_path = os.path.join(files_path, item)
        if not check_performance:
            poly = True
            attrs = None
            test_single_file(file_path, attrs, poly, profiling=False)
        else:
            file_name = item.split('.')[0]
            file_result = os.path.join(output, file_name + ".csv")
            if os.path.isfile(file_result):
                os.remove(file_result)
            if subprocess.call("nvprof --csv --log-file %s python3 %s -f %s" %
                               (file_result, script_file, file_path), shell=True):
                raise ValueError("Test %s failed" % file_path)
            if not compare_base_line(pwd, file_result, network, level, file_name):
                raise ValueError("Performance degradation of %s!" % file_path)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_alexnet_gpu_level0():
    test_network("gpu", "alexnet", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_alexnet_gpu_level0_perf():
    test_network("gpu", "alexnet", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_alexnet_gpu_level1_tensorcore():
    test_network("gpu", "alexnet", "level1_tensorcore", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_15classifier_gpu_level0():
    test_network("gpu", "bert_15classifier", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_15classifier_gpu_level0_perf():
    test_network("gpu", "bert_15classifier", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_15classifier_gpu_level1():
    test_network("gpu", "bert_15classifier", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_base_gpu_level0():
    test_network("gpu", "bert_base", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_base_gpu_level0_perf():
    test_network("gpu", "bert_base", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_base_gpu_level1():
    test_network("gpu", "bert_base", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_large_gpu_level0():
    test_network("gpu", "bert_large", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_large_gpu_level0_perf():
    test_network("gpu", "bert_large", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bert_large_gpu_level1():
    test_network("gpu", "bert_large", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cyclegan_gpu_level0():
    test_network("gpu", "cyclegan", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cyclegan_gpu_level0_perf():
    test_network("gpu", "cyclegan", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_cyclegan_gpu_level1():
    test_network("gpu", "cyclegan", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deep_fm_gpu_level0():
    test_network("gpu", "deep_fm", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deep_fm_gpu_level0_perf():
    test_network("gpu", "deep_fm", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deep_fm_gpu_level1():
    test_network("gpu", "deep_fm", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deep_speech_gpu_level0():
    test_network("gpu", "deep_speech", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deep_speech_gpu_level0_perf():
    test_network("gpu", "deep_speech", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_deep_speech_gpu_level1():
    test_network("gpu", "deep_speech", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dqn_gpu_level0():
    test_network("gpu", "dqn", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_dqn_gpu_level0_perf():
    test_network("gpu", "dqn", "level0", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_efficientnet_gpu_level0_test0():
    test_network("gpu", "efficientnet", "level0", 3, 0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_efficientnet_gpu_level0_test0_perf():
    test_network("gpu", "efficientnet", "level0", 3, 0, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_efficientnet_gpu_level0_test1():
    test_network("gpu", "efficientnet", "level0", 3, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_efficientnet_gpu_level0_test1_perf():
    test_network("gpu", "efficientnet", "level0", 3, 1, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_efficientnet_gpu_level0_test2():
    test_network("gpu", "efficientnet", "level0", 3, 2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_efficientnet_gpu_level0_test2_perf():
    test_network("gpu", "efficientnet", "level0", 3, 2, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_efficientnet_gpu_level1_test0():
    test_network("gpu", "efficientnet", "level1", 2, 0, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_efficientnet_gpu_level1_test1():
    test_network("gpu", "efficientnet", "level1", 2, 1, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gpt_gpu_level0():
    test_network("gpu", "gpt", "level0")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_custom_gpu_level0():
    test_network("gpu", "custom", "level0")


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_googlenet_gpu_level0():
    test_network("gpu", "googlenet", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_googlenet_gpu_level0_perf():
    test_network("gpu", "googlenet", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_googlenet_gpu_level1():
    test_network("gpu", "googlenet", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inception_v3_gpu_level0_test0():
    test_network("gpu", "inception_v3", "level0", 2, 0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inception_v3_gpu_level0_test0_perf():
    test_network("gpu", "inception_v3", "level0", 2, 0, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inception_v3_gpu_level0_test1():
    test_network("gpu", "inception_v3", "level0", 2, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inception_v3_gpu_level0_test1_perf():
    test_network("gpu", "inception_v3", "level0", 2, 1, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inception_v3_gpu_level1_test0():
    test_network("gpu", "inception_v3", "level1", 2, 0, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_inception_v3_gpu_level1_test1():
    test_network("gpu", "inception_v3", "level1", 2, 1, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lenet_gpu_level0():
    test_network("gpu", "lenet", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lenet_gpu_level0_perf():
    test_network("gpu", "lenet", "level0", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lenet_quant_gpu_level0():
    test_network("gpu", "lenet_quant", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lenet_quant_gpu_level0_perf():
    test_network("gpu", "lenet_quant", "level0", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lstm_gpu_level0():
    test_network("gpu", "lstm", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lstm_gpu_level0_perf():
    test_network("gpu", "lstm", "level0", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_gpu_level0():
    test_network("gpu", "mobilenet_v2", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_gpu_level0_perf():
    test_network("gpu", "mobilenet_v2", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_gpu_level1():
    test_network("gpu", "mobilenet_v2", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_quant_gpu_level0_test0():
    test_network("gpu", "mobilenet_v2_quant", "level0", 2, 0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_quant_gpu_level0_test0_perf():
    test_network("gpu", "mobilenet_v2_quant", "level0", 2, 0, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_quant_gpu_level0_test1():
    test_network("gpu", "mobilenet_v2_quant", "level0", 2, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_quant_gpu_level0_test1_perf():
    test_network("gpu", "mobilenet_v2_quant", "level0", 2, 1, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v2_quant_gpu_level1():
    test_network("gpu", "mobilenet_v2_quant", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v3_gpu_level0():
    test_network("gpu", "mobilenet_v3", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v3_gpu_level0_perf():
    test_network("gpu", "mobilenet_v3", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_mobilenet_v3_gpu_level1():
    test_network("gpu", "mobilenet_v3", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ocean_model_gpu_level0():
    test_network("gpu", "ocean_model", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ocean_model_gpu_level0_perf():
    test_network("gpu", "ocean_model", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ocean_model_gpu_level1_test0():
    test_network("gpu", "ocean_model", "level1", 2, 0, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ocean_model_gpu_level1_test1():
    test_network("gpu", "ocean_model", "level1", 2, 1, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resnet101_gpu_level0():
    test_network("gpu", "resnet101", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resnet101_gpu_level0_perf():
    test_network("gpu", "resnet101", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resnet101_gpu_level1():
    test_network("gpu", "resnet101", "level1", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_resnet101_gpu_level1_tensorcore():
    test_network("gpu", "resnet101", "level1_tensorcore", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_retinaface_resnet50_gpu_level0():
    test_network("gpu", "retinaface_resnet50", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_retinaface_resnet50_gpu_level0_perf():
    test_network("gpu", "retinaface_resnet50", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_retinaface_resnet50_gpu_level1():
    test_network("gpu", "retinaface_resnet50", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ssd_gpu_level0():
    test_network("gpu", "ssd", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ssd_gpu_level0_perf():
    test_network("gpu", "ssd", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ssd_gpu_level1():
    test_network("gpu", "ssd", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_base_gpu_level0():
    test_network("gpu", "tinybert_base", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_base_gpu_level0_perf():
    test_network("gpu", "tinybert_base", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_base_gpu_level1():
    test_network("gpu", "tinybert_base", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_mnli_gpu_level0():
    test_network("gpu", "tinybert_mnli", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_mnli_gpu_level0_perf():
    test_network("gpu", "tinybert_mnli", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_mnli_gpu_level1():
    test_network("gpu", "tinybert_mnli", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_qnli_gpu_level0():
    test_network("gpu", "tinybert_qnli", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_qnli_gpu_level0_perf():
    test_network("gpu", "tinybert_qnli", "level0", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_sst2_gpu_level0():
    test_network("gpu", "tinybert_sst2", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_sst2_gpu_level0_perf():
    test_network("gpu", "tinybert_sst2", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_tinybert_sst2_gpu_level1():
    test_network("gpu", "tinybert_sst2", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transformer_gpu_level0():
    test_network("gpu", "transformer", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transformer_gpu_level0_perf():
    test_network("gpu", "transformer", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transformer_gpu_level1_test0():
    test_network("gpu", "transformer", "level1", 6, 0, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transformer_gpu_level1_test1():
    test_network("gpu", "transformer", "level1", 6, 1, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transformer_gpu_level1_test2():
    test_network("gpu", "transformer", "level1", 6, 2, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transformer_gpu_level1_test3():
    test_network("gpu", "transformer", "level1", 6, 3, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transformer_gpu_level1_test4():
    test_network("gpu", "transformer", "level1", 6, 4, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_transformer_gpu_level1_test5():
    test_network("gpu", "transformer", "level1", 6, 5, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vgg16_gpu_level0():
    test_network("gpu", "vgg16", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vgg16_gpu_level0_perf():
    test_network("gpu", "vgg16", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_vgg16_gpu_level1():
    test_network("gpu", "vgg16", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_warpctc_gpu_level0():
    test_network("gpu", "warpctc", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_warpctc_gpu_level0_perf():
    test_network("gpu", "warpctc", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_warpctc_gpu_level1():
    test_network("gpu", "warpctc", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_gpu_level0():
    test_network("gpu", "wide_deep", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_gpu_level0_perf():
    test_network("gpu", "wide_deep", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_gpu_level1():
    test_network("gpu", "wide_deep", "level1", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_gpu_level1_tensorcore():
    test_network("gpu", "wide_deep", "level1_tensorcore", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_ps_gpu_level0():
    test_network("gpu", "wide_deep_ps", "level0")


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_ps_gpu_level0_perf():
    test_network("gpu", "wide_deep_ps", "level0", check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_wide_deep_ps_gpu_level1():
    test_network("gpu", "wide_deep_ps", "level1", check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level0_test0():
    test_network("gpu", "yolov3_darknet53", "level0", 2, 0)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level0_test0_perf():
    test_network("gpu", "yolov3_darknet53", "level0", 2, 0, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level0_test1():
    test_network("gpu", "yolov3_darknet53", "level0", 2, 1)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level0_test1_perf():
    test_network("gpu", "yolov3_darknet53", "level0", 2, 1, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level1_test0():
    test_network("gpu", "yolov3_darknet53", "level1", 2, 0, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level1_test1():
    test_network("gpu", "yolov3_darknet53", "level1", 2, 1, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level1_tensorcore_test0():
    test_network("gpu", "yolov3_darknet53", "level1_tensorcore", 2, 0, check_performance=True)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_yolov3_darknet53_gpu_level1_tensorcore_test1():
    test_network("gpu", "yolov3_darknet53", "level1_tensorcore", 2, 1, check_performance=True)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_transformer_ascend_level0():
    test_network("ascend", "transformer", "level0")
