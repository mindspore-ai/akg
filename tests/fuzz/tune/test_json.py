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

"""test composite json tuning"""
from autotuning.job import launch_json

json_dir_name = "bn_tune"
debug_mode_ = True
save_res_ = True
launch_json(debug_mode=debug_mode_, save_res=save_res_, json_input_dir=json_dir_name)
