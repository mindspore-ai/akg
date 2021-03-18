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

"""test"""
import time
from tests.fuzz.tune.autotuning.job import launch
from tests.common.test_run.sub_run import sub_execute

time_start = time.time()
op_type_ = 'sub'
debug_mode_ = True
save_res_ = True
all_space_ = False
desc_ = ('024_sub_64_16_128_128_64_16_128_128_fp16', sub_execute, [(64, 16, 128, 128), (64, 16, 128, 1), 'float16'])
launch(op_type=op_type_, debug_mode=debug_mode_, save_res=save_res_, desc=desc_, all_space=all_space_)
time_end = time.time()
print("launch time: ", time_end - time_start)
