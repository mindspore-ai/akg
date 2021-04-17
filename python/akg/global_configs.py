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

"""global_configs"""
import akg.tvm

CUDA_META_PATH = './cuda_meta/'
ASCEND_META_PATH = './kernel_meta/'

DUMP_IR_FLAG = 'MS_AKG_DUMP_IR'
DUMP_CODE_FLAG = 'MS_AKG_DUMP_CODE'

@akg.tvm.register_func
def get_cuda_meta_path():
  return CUDA_META_PATH

@akg.tvm.register_func
def get_ascend_meta_path():
  return ASCEND_META_PATH

@akg.tvm.register_func
def get_dump_ir_flag():
  return DUMP_IR_FLAG

@akg.tvm.register_func
def get_dump_code_flag():
  return DUMP_CODE_FLAG
